"""AgentBridge adapter plugin for Strands Agents."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CompiledStrandsAgent:
    """Compiled Strands agent plus the source AgentBridge spec."""

    spec: AgentSpec
    native_agent: Any
    native_tools: list[Any]
    config: dict[str, Any]
    sdk: Any


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "strands"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "tools.mcp": "extension",
                "structured_output": "full",
                "guardrails": "extension",
                "observability.tracing": "extension",
                "deployment.serverless": "extension",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec instructions to Strands system_prompt.",
                "agent.model": "Passes model strings through to Strands; provider compatibility is Strands/model dependent.",
                "tools.sync": "Maps ToolSpec callables to Strands @tool wrappers.",
                "tools.mcp": "Appends native MCP client/tool-provider objects from StrandsExtension.mcp_clients; string labels remain metadata-only hints.",
                "structured_output": "Maps AgentSpec.output_type to Strands structured_output_model and validates native structured output.",
                "guardrails": "Records guardrail/intervention hints and forwards native interventions when supplied.",
                "observability.tracing": "Passes trace_attributes and preserves extension/runtime summaries.",
                "deployment.serverless": "Records deployment_target metadata for AgentCore/Lambda-style deployment paths.",
                "streaming.events": "Uses Strands stream_async when available and normalizes events best-effort.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledStrandsAgent:
        """Translate AgentSpec into a Strands Agent."""

        sdk = _load_strands_sdk()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        native_tools = [
            *_to_strands_tools(sdk, spec.tools),
            *_native_mcp_clients(config),
        ]
        agent_kwargs: dict[str, Any] = {
            "name": spec.name,
            "model": _model_for_spec(spec),
            "system_prompt": spec.instructions,
            "tools": native_tools,
        }
        if spec.output_type is not None:
            agent_kwargs["structured_output_model"] = spec.output_type
        _copy_native_agent_options(
            config,
            agent_kwargs,
            (
                "conversation_manager",
                "context_manager",
                "trace_attributes",
                "hooks",
                "plugins",
                "interventions",
                "session_manager",
                "memory_manager",
                "tool_executor",
                "retry_strategy",
                "checkpointing",
                "sandbox",
                "storage",
                "background_tasks",
                "agent_id",
                "description",
                "structured_output_prompt",
                "load_tools_from_directory",
                "record_direct_tool_call",
            ),
        )
        if config.get("metadata"):
            agent_kwargs["state"] = {"metadata": config["metadata"]}
        if spec.model == "agentbridge/offline":
            agent_kwargs["callback_handler"] = None

        native_agent = sdk.Agent(**agent_kwargs)
        return CompiledStrandsAgent(
            spec=spec,
            native_agent=native_agent,
            native_tools=native_tools,
            config=config,
            sdk=sdk,
        )

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled Strands agent and return a normalized result."""

        compiled_agent = _ensure_compiled(compiled)
        result = compiled_agent.native_agent(
            run_input.input,
            invocation_state=_invocation_state(run_input),
        )
        output = _final_output(result)
        events = _events_from_messages(
            getattr(compiled_agent.native_agent, "messages", []),
            backend=self.backend_name,
        )
        events.extend(
            [
                AgentEvent(
                    type="message",
                    backend=self.backend_name,
                    data={
                        "content": output,
                        "agent": compiled_agent.spec.name,
                        "stop_reason": getattr(result, "stop_reason", None),
                    },
                ),
                AgentEvent(
                    type="complete",
                    backend=self.backend_name,
                    data={"output": output},
                ),
            ]
        )
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            usage=_usage_from_result(result),
            metadata={
                "agent": compiled_agent.spec.name,
                "interrupts": _safe_repr(getattr(result, "interrupts", None)),
                "checkpoint": _safe_repr(getattr(result, "checkpoint", None)),
                "invocation_state": _safe_summary(_invocation_state(run_input)),
                "extension_config": _safe_summary(compiled_agent.config),
                "extension_summary": _extension_summary(compiled_agent.config),
                "native_agent_type": type(compiled_agent.native_agent).__name__,
            },
            raw=result,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        if _uses_offline_model(compiled_agent):
            yield from self.run(compiled_agent, run_input).events
            return

        if not hasattr(compiled_agent.native_agent, "stream_async"):
            yield from self.run(compiled_agent, run_input).events
            return

        yielded = False
        for native_event in _run_async_stream(compiled_agent, run_input):
            yielded = True
            yield _normalize_native_event(native_event, backend=self.backend_name)
        if not yielded:
            yield from self.run(compiled_agent, run_input).events


def _load_strands_sdk() -> Any:
    try:
        return import_module("strands")
    except ImportError as exc:  # pragma: no cover - message assertion path.
        raise ImportError(
            "Strands Agents is not installed. Install this plugin with `strands-agents>=1.55,<2`."
        ) from exc


def _to_strands_tool(sdk: Any, tool_spec: Any) -> Any:
    return sdk.tool(
        tool_spec.handler,
        name=tool_spec.name,
        description=tool_spec.description,
        inputSchema=tool_spec.input_schema,
    )


def _to_strands_tools(sdk: Any, tool_specs: list[Any]) -> list[Any]:
    return [_to_strands_tool(sdk, tool) for tool in tool_specs]


def _native_mcp_clients(config: dict[str, Any]) -> list[Any]:
    clients: list[Any] = []
    for client in config.get("mcp_clients") or []:
        if isinstance(client, str):
            continue
        clients.append(client)
    return clients


def _copy_native_agent_options(
    config: dict[str, Any],
    agent_kwargs: dict[str, Any],
    option_names: tuple[str, ...],
) -> None:
    for option_name in option_names:
        if option_name not in config:
            continue
        value = config[option_name]
        if value is None:
            continue
        if option_name in {"checkpointing", "load_tools_from_directory", "record_direct_tool_call"}:
            agent_kwargs[option_name] = bool(value)
        else:
            agent_kwargs[option_name] = value


def _model_for_spec(spec: AgentSpec) -> Any:
    if spec.model == "agentbridge/offline":
        model_base = import_module("strands.models.model").Model

        class AgentBridgeOfflineModel(_AgentBridgeOfflineModelBase, model_base):
            pass

        return AgentBridgeOfflineModel()
    return spec.model


class _AgentBridgeOfflineModelBase:
    """No-network Strands model used by conformance tests."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "model_id": "agentbridge/offline",
            "context_window_limit": 4096,
        }

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def get_config(self) -> Any:
        return dict(self._config)

    async def structured_output(
        self,
        output_model: type[Any],
        prompt: Any,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        del prompt, system_prompt, kwargs
        yield {"output": output_model()}

    async def stream(
        self,
        messages: Any,
        tool_specs: list[Any] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: Any | None = None,
        system_prompt_content: list[Any] | None = None,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        del system_prompt, tool_choice, system_prompt_content, invocation_state, cancel_signal
        del kwargs

        tool_result = _latest_tool_result(messages)
        if tool_result is not None:
            async for event in self._text_stream(f"offline tool result: {tool_result}"):
                yield event
            return

        if tool_specs:
            tool_spec = tool_specs[0]
            tool_name = str(tool_spec.get("name", "tool"))
            async for event in self._tool_use_stream(
                name=tool_name,
                arguments=_arguments_for_tool_spec(tool_spec, messages),
            ):
                yield event
            return

        async for event in self._text_stream(f"offline response: {_last_user_text(messages)}"):
            yield event

    async def _text_stream(self, text: str) -> Any:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}}

    async def _tool_use_stream(self, *, name: str, arguments: dict[str, Any]) -> Any:
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "agentbridge-offline-tool-use",
                        "name": name,
                    }
                }
            }
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(arguments)}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}}


def _uses_offline_model(compiled: CompiledStrandsAgent) -> bool:
    return isinstance(getattr(compiled.native_agent, "model", None), _AgentBridgeOfflineModelBase)


def _arguments_for_tool_spec(tool_spec: dict[str, Any], messages: Any) -> dict[str, Any]:
    schema = tool_spec.get("inputSchema") or {}
    schema = schema.get("json", schema)
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    if not properties:
        return {"input": _last_user_text(messages)}

    arguments: dict[str, Any] = {}
    selected = list(required) or list(properties)
    for name in selected:
        property_schema = properties.get(name, {})
        if "default" in property_schema:
            arguments[name] = property_schema["default"]
        else:
            arguments[name] = _fallback_value_for_schema(property_schema, messages)
    return arguments


def _fallback_value_for_schema(property_schema: dict[str, Any], messages: Any) -> Any:
    raw_type = property_schema.get("type", "string")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    if "boolean" in types:
        return True
    if "integer" in types:
        return 1
    if "number" in types:
        return 1.0
    if "array" in types:
        return []
    if "object" in types:
        return {}
    return _last_user_text(messages)


def _latest_tool_result(messages: Any) -> Any | None:
    for message in reversed(messages or []):
        for block in reversed(message.get("content", []) if isinstance(message, dict) else []):
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            content = block["toolResult"].get("content", [])
            if content and isinstance(content[0], dict):
                return content[0].get("text", content[0])
            return content
    return None


def _last_user_text(messages: Any) -> str:
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        parts: list[str] = []
        for block in message.get("content", []):
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        if parts:
            return " ".join(parts)
    return ""


def _events_from_messages(messages: Any, *, backend: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                events.append(
                    AgentEvent(
                        type="tool_call",
                        backend=backend,
                        data=block["toolUse"],
                    )
                )
            if "toolResult" in block:
                events.append(
                    AgentEvent(
                        type="tool_result",
                        backend=backend,
                        data=block["toolResult"],
                    )
                )
    return events


def _ensure_compiled(compiled: Any) -> CompiledStrandsAgent:
    if not isinstance(compiled, CompiledStrandsAgent):
        raise TypeError("Strands adapter expected CompiledStrandsAgent from compile().")
    return compiled


def _invocation_state(run_input: RunInput) -> dict[str, Any]:
    state = dict(run_input.context)
    if run_input.session_id:
        state["session_id"] = run_input.session_id
    if run_input.metadata:
        state["metadata"] = run_input.metadata
    return state


def _extension_summary(config: dict[str, Any]) -> dict[str, Any]:
    native_agent_options = [
        "conversation_manager",
        "context_manager",
        "trace_attributes",
        "hooks",
        "plugins",
        "interventions",
        "session_manager",
        "memory_manager",
        "tool_executor",
        "retry_strategy",
        "checkpointing",
        "sandbox",
        "storage",
        "background_tasks",
        "agent_id",
        "description",
        "structured_output_prompt",
        "load_tools_from_directory",
        "record_direct_tool_call",
    ]
    applied_native_options = [
        option_name
        for option_name in native_agent_options
        if option_name in config and config[option_name] is not None
    ]
    return {
        "conversation_manager": "conversation_manager" in applied_native_options,
        "context_manager": "context_manager" in applied_native_options,
        "hooks_count": len(config.get("hooks") or []),
        "plugins_count": len(config.get("plugins") or []),
        "interventions_count": len(config.get("interventions") or []),
        "mcp_clients": _safe_summary(config.get("mcp_clients") or []),
        "native_mcp_clients_count": len(_native_mcp_clients(config)),
        "trace_attributes": _safe_summary(config.get("trace_attributes") or {}),
        "guardrails": _safe_summary(config.get("guardrails") or []),
        "deployment_target": config.get("deployment_target"),
        "session_manager": "session_manager" in applied_native_options,
        "memory_manager": "memory_manager" in applied_native_options,
        "applied_native_options": applied_native_options,
    }


def _final_output(result: Any) -> Any:
    structured_output = getattr(result, "structured_output", None)
    if structured_output is not None:
        return structured_output
    message = getattr(result, "message", None)
    if message is None:
        return result
    return _message_to_text(message)


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", message)
    else:
        content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", item))))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "".join(parts)
    return str(content)


def _usage_from_result(result: Any) -> dict[str, Any]:
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return {}
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    if isinstance(metrics, dict):
        return metrics
    return {"raw": repr(metrics)}


def _run_async_stream(compiled: CompiledStrandsAgent, run_input: RunInput) -> list[Any]:
    async def collect() -> list[Any]:
        events: list[Any] = []
        async for event in compiled.native_agent.stream_async(
            run_input.input,
            invocation_state=_invocation_state(run_input),
        ):
            events.append(event)
        return events

    try:
        return asyncio.run(collect())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(collect())
        finally:
            loop.close()


def _normalize_native_event(native_event: Any, *, backend: str) -> AgentEvent:
    payload = _payload(native_event)
    if isinstance(native_event, dict):
        keys = {str(key).lower() for key in native_event}
    else:
        keys = {type(native_event).__name__.lower()}
    if any("tool" in key and ("result" in key or "output" in key) for key in keys):
        event_type = "tool_result"
    elif any("tool" in key for key in keys):
        event_type = "tool_call"
    elif any("error" in key or "exception" in key for key in keys):
        event_type = "error"
    elif any("event" in key or "delta" in key or "message" in key for key in keys):
        event_type = "message"
    else:
        event_type = "workflow"
    payload.setdefault("native_type", type(native_event).__name__)
    return AgentEvent(type=event_type, backend=backend, data=payload)


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _safe_repr(value: Any) -> str | None:
    if value is None:
        return None
    return repr(value)


def _safe_summary(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    return type(value).__name__
