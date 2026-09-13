"""AgentBridge adapter plugin for OpenAI Agents SDK."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CompiledOpenAIAgentsAgent:
    """Compiled OpenAI Agents SDK objects plus the source AgentBridge spec."""

    spec: AgentSpec
    native_agent: Any
    native_tools: list[Any]
    config: dict[str, Any]
    sdk: Any


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "openai_agents"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "structured_output": "full",
                "workflow.handoffs": "extension",
                "guardrails": "extension",
                "human_approval": "extension",
                "observability.tracing": "extension",
                "observability.diagnostics": "full",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec name/instructions/model/tools to SDK Agent.",
                "agent.model": "Passes LiteLLM-style model strings through to the SDK; provider compatibility is SDK/model dependent.",
                "tools.sync": "Maps ToolSpec callables to OpenAI Agents function_tool wrappers.",
                "structured_output": "Passes AgentSpec.output_type to SDK Agent output_type and validates typed final_output.",
                "workflow.handoffs": "Forwards native handoffs and handoff metadata through OpenAIAgentsExtension.",
                "guardrails": "Forwards native input/output guardrails and records approval policy hints.",
                "human_approval": "Records approval_policy metadata; native approval/resume flow tests are not implemented yet.",
                "observability.tracing": "Preserves trace/run config summaries; SDK/provider tracing remains extension-level.",
                "observability.diagnostics": "Normalizes approval interruptions, guardrail results, response IDs, and resumability hints into run_diagnostics metadata.",
                "streaming.events": "Uses SDK run_streamed when available and normalizes streamed event objects best-effort.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledOpenAIAgentsAgent:
        """Translate AgentSpec into OpenAI Agents SDK objects."""

        sdk = _load_agents_sdk()
        native_tools = [_to_function_tool(sdk, tool) for tool in spec.tools]
        config = dict(spec.backend_config.get(self.backend_name, {}))
        agent_kwargs: dict[str, Any] = {
            "name": spec.name,
            "instructions": spec.instructions,
            "model": _model_for_spec(sdk, spec),
            "tools": native_tools,
        }
        if spec.output_type is not None:
            agent_kwargs["output_type"] = spec.output_type
        _copy_native_agent_options(
            config,
            agent_kwargs,
            (
                "handoff_description",
                "handoffs",
                "mcp_servers",
                "mcp_config",
                "prompt",
                "model_settings",
                "input_guardrails",
                "output_guardrails",
                "hooks",
                "tool_use_behavior",
                "reset_tool_choice",
            ),
        )
        if config.get("guardrails") and "input_guardrails" not in agent_kwargs:
            agent_kwargs["input_guardrails"] = config["guardrails"]
        if config.get("metadata"):
            agent_kwargs["metadata"] = config["metadata"]

        try:
            native_agent = sdk.Agent(**agent_kwargs)
        except TypeError:
            # Older/fake SDKs may not accept extension-only metadata yet.
            agent_kwargs.pop("metadata", None)
            native_agent = sdk.Agent(**agent_kwargs)

        return CompiledOpenAIAgentsAgent(
            spec=spec,
            native_agent=native_agent,
            native_tools=native_tools,
            config=config,
            sdk=sdk,
        )

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled agent through OpenAI Agents Runner.run_sync."""

        compiled_agent = _ensure_compiled(compiled)
        runner_kwargs = _runner_kwargs(compiled_agent, run_input)
        result = compiled_agent.sdk.Runner.run_sync(
            compiled_agent.native_agent,
            run_input.input,
            **runner_kwargs,
        )
        output = _final_output(result)
        events = _events_from_result(result, backend=self.backend_name)
        if not events:
            events = [
                AgentEvent(
                    type="message",
                    backend=self.backend_name,
                    data={"content": output, "agent": compiled_agent.spec.name},
                )
            ]
        events.append(
            AgentEvent(
                type="complete",
                backend=self.backend_name,
                data={"output": output},
            )
        )
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            usage=_usage_from_result(result),
            metadata={
                "agent": compiled_agent.spec.name,
                "runner_kwargs": _safe_summary(runner_kwargs),
                "extension_config": _safe_summary(compiled_agent.config),
                "extension_summary": _extension_summary(compiled_agent.config, runner_kwargs),
                "native_agent_type": type(compiled_agent.native_agent).__name__,
                "run_diagnostics": _run_diagnostics(result),
            },
            raw=result,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        if _uses_offline_model(compiled_agent):
            yield from self.run(compiled_agent, run_input).events
            return

        if not hasattr(compiled_agent.sdk.Runner, "run_streamed"):
            yield from self.run(compiled_agent, run_input).events
            return

        stream_result = compiled_agent.sdk.Runner.run_streamed(
            compiled_agent.native_agent,
            run_input.input,
            **_runner_kwargs(compiled_agent, run_input),
        )
        yielded = False
        stream_events = _stream_events(stream_result)
        for native_event in stream_events:
            yielded = True
            yield _normalize_native_event(native_event, backend=self.backend_name)

        output = _final_output(stream_result)
        if output is not None:
            yielded = True
            yield AgentEvent(
                type="complete",
                backend=self.backend_name,
                data={"output": output},
            )

        if not yielded:
            yield from self.run(compiled_agent, run_input).events


def _load_agents_sdk() -> Any:
    try:
        return import_module("agents")
    except ImportError as exc:  # pragma: no cover - exercised through message assertions.
        raise ImportError(
            "OpenAI Agents SDK is not installed or cannot resolve with the current environment. "
            "Install this plugin in an isolated environment with `openai-agents>=0.22,<1`. "
            "Known nuance: openai-agents 0.22.x requires openai>=3, while some LiteLLM "
            "versions require openai<3."
        ) from exc


def _to_function_tool(sdk: Any, tool: Any) -> Any:
    return sdk.function_tool(
        tool.handler,
        name_override=tool.name,
        description_override=tool.description,
    )


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
        if option_name == "reset_tool_choice":
            agent_kwargs[option_name] = bool(value)
        else:
            agent_kwargs[option_name] = value


def _model_for_spec(sdk: Any, spec: AgentSpec) -> Any:
    if spec.model == "agentbridge/offline":

        class AgentBridgeOfflineModel(_AgentBridgeOfflineModelBase, sdk.Model):
            pass

        return AgentBridgeOfflineModel(sdk)
    return spec.model


class _AgentBridgeOfflineModelBase:
    """No-network OpenAI Agents SDK model used by conformance tests."""

    def __init__(self, sdk: Any):
        self._sdk = sdk

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> Any:
        del system_instructions, model_settings, handoffs, tracing
        del previous_response_id, conversation_id, prompt

        items = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
        tool_output = _latest_tool_output(items)
        if tool_output is not None:
            return self._message_response(f"offline tool result: {tool_output}")
        if output_schema is not None:
            return self._message_response(_structured_output_json(output_schema, items))
        if tools:
            tool = tools[0]
            argument_name = _first_tool_argument_name(tool)
            return self._tool_call_response(
                name=getattr(tool, "name", "tool"),
                arguments={argument_name: _last_user_text(items)},
            )
        return self._message_response(f"offline response: {_last_user_text(items)}")

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> Any:
        del system_instructions, input, model_settings, tools, output_schema, handoffs, tracing
        del previous_response_id, conversation_id, prompt
        if False:
            yield None

    def _message_response(self, text: str) -> Any:
        items = import_module("agents.items")
        usage = import_module("agents.usage")
        responses = import_module("openai.types.responses")
        message = responses.ResponseOutputMessage(
            id="agentbridge-offline-message",
            content=[
                responses.ResponseOutputText(
                    annotations=[],
                    text=text,
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        return items.ModelResponse(
            output=[message],
            usage=usage.Usage(requests=1),
            response_id="agentbridge-offline-response",
        )

    def _tool_call_response(self, *, name: str, arguments: dict[str, Any]) -> Any:
        items = import_module("agents.items")
        usage = import_module("agents.usage")
        responses = import_module("openai.types.responses")
        tool_call = responses.ResponseFunctionToolCall(
            arguments=json.dumps(arguments),
            call_id="agentbridge-offline-tool-call",
            name=name,
            type="function_call",
            id="agentbridge-offline-tool-call",
            status="completed",
        )
        return items.ModelResponse(
            output=[tool_call],
            usage=usage.Usage(requests=1),
            response_id="agentbridge-offline-response",
        )


def _uses_offline_model(compiled: CompiledOpenAIAgentsAgent) -> bool:
    return isinstance(getattr(compiled.native_agent, "model", None), _AgentBridgeOfflineModelBase)


def _first_tool_argument_name(tool: Any) -> str:
    schema = getattr(tool, "params_json_schema", {}) or {}
    required = schema.get("required") or []
    if required:
        return str(required[0])
    properties = schema.get("properties") or {}
    if properties:
        return str(next(iter(properties)))
    return "input"


def _structured_output_json(output_schema: Any, items: list[Any]) -> str:
    schema = _output_json_schema(output_schema)
    return json.dumps(_value_for_schema(schema, items))


def _output_json_schema(output_schema: Any) -> dict[str, Any]:
    json_schema = getattr(output_schema, "json_schema", None)
    if callable(json_schema):
        schema = json_schema()
        if isinstance(schema, dict):
            return schema
    schema = getattr(output_schema, "_output_schema", None)
    if isinstance(schema, dict):
        return schema
    output_type = getattr(output_schema, "output_type", None)
    model_json_schema = getattr(output_type, "model_json_schema", None)
    if callable(model_json_schema):
        schema = model_json_schema()
        if isinstance(schema, dict):
            return schema
    return {"type": "object"}


def _value_for_schema(schema: dict[str, Any], items: list[Any]) -> Any:
    if "default" in schema:
        return schema["default"]
    raw_type = schema.get("type", "string")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    if "object" in types:
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        selected = list(required) or list(properties)
        return {
            name: _value_for_schema(properties.get(name, {}), items)
            for name in selected
        }
    if "array" in types:
        return []
    if "boolean" in types:
        return True
    if "integer" in types:
        return 1
    if "number" in types:
        return 1.0
    return _last_user_text(items) or "ok"


def _latest_tool_output(items: list[Any]) -> Any | None:
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            return item.get("output")
    return None


def _last_user_text(items: list[Any]) -> str:
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {"input_text", "text"}
                ]
                if text_parts:
                    return " ".join(text_parts)
    return ""


def _ensure_compiled(compiled: Any) -> CompiledOpenAIAgentsAgent:
    if not isinstance(compiled, CompiledOpenAIAgentsAgent):
        raise TypeError("OpenAI Agents adapter expected CompiledOpenAIAgentsAgent from compile().")
    return compiled


def _runner_kwargs(compiled: CompiledOpenAIAgentsAgent, run_input: RunInput) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    config = compiled.config
    if "context" in config:
        kwargs["context"] = config["context"]
    if "max_turns" in config:
        kwargs["max_turns"] = config["max_turns"]
    if "run_hooks" in config:
        kwargs["hooks"] = config["run_hooks"]
    if "error_handlers" in config:
        kwargs["error_handlers"] = config["error_handlers"]
    if "previous_response_id" in config:
        kwargs["previous_response_id"] = config["previous_response_id"]
    if "auto_previous_response_id" in config:
        kwargs["auto_previous_response_id"] = bool(config["auto_previous_response_id"])
    if run_input.session_id:
        kwargs["conversation_id"] = run_input.session_id
    elif config.get("conversation_id"):
        kwargs["conversation_id"] = config["conversation_id"]
    elif config.get("session_id"):
        kwargs["conversation_id"] = config["session_id"]
    if "session" in config:
        kwargs["session"] = config["session"]
    if config.get("run_config"):
        kwargs["run_config"] = config["run_config"]
    elif _uses_offline_model(compiled) and hasattr(compiled.sdk, "RunConfig"):
        kwargs["run_config"] = compiled.sdk.RunConfig(tracing_disabled=True)
    return kwargs


def _extension_summary(config: dict[str, Any], runner_kwargs: dict[str, Any]) -> dict[str, Any]:
    native_agent_options = [
        "handoff_description",
        "handoffs",
        "mcp_servers",
        "mcp_config",
        "prompt",
        "model_settings",
        "input_guardrails",
        "output_guardrails",
        "guardrails",
        "hooks",
        "tool_use_behavior",
        "reset_tool_choice",
    ]
    native_runner_options = [
        "context",
        "max_turns",
        "run_hooks",
        "run_config",
        "error_handlers",
        "previous_response_id",
        "auto_previous_response_id",
        "conversation_id",
        "session_id",
        "session",
    ]
    return {
        "handoffs_count": len(config.get("handoffs") or []),
        "mcp_servers_count": len(config.get("mcp_servers") or []),
        "guardrails_count": len(config.get("guardrails") or []),
        "input_guardrails_count": len(config.get("input_guardrails") or []),
        "output_guardrails_count": len(config.get("output_guardrails") or []),
        "approval_policy": _safe_summary(config.get("approval_policy") or {}),
        "tracing": bool(config.get("tracing", False)),
        "metadata": _safe_summary(config.get("metadata") or {}),
        "conversation_id": runner_kwargs.get("conversation_id"),
        "runner_options": sorted(runner_kwargs),
        "applied_native_agent_options": [
            option_name
            for option_name in native_agent_options
            if option_name in config and config[option_name] is not None
        ],
        "applied_native_runner_options": [
            option_name
            for option_name in native_runner_options
            if option_name in config and config[option_name] is not None
        ],
    }


def _final_output(result: Any) -> Any:
    return getattr(result, "final_output", result)


def _usage_from_result(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {"raw": repr(usage)}


def _events_from_result(result: Any, *, backend: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for item in getattr(result, "new_items", []) or []:
        events.append(_normalize_native_event(item, backend=backend))
    for interruption in getattr(result, "interruptions", []) or []:
        events.append(
            AgentEvent(
                type="workflow",
                backend=backend,
                data={
                    "event": "approval_required",
                    "interruption": _safe_summary(interruption),
                    "native_type": type(interruption).__name__,
                },
            )
        )
    for guardrail_kind, guardrail_results in _guardrail_result_groups(result).items():
        for guardrail_result in guardrail_results:
            events.append(
                AgentEvent(
                    type="workflow",
                    backend=backend,
                    data={
                        "event": "guardrail_result",
                        "kind": guardrail_kind,
                        "result": _safe_summary(guardrail_result),
                        "native_type": type(guardrail_result).__name__,
                    },
                )
            )
    return events


def _stream_events(stream_result: Any) -> Iterator[Any]:
    if hasattr(stream_result, "stream_events"):
        yield from stream_result.stream_events()
    elif hasattr(stream_result, "__iter__"):
        yield from stream_result


def _normalize_native_event(native_event: Any, *, backend: str) -> AgentEvent:
    event_name = type(native_event).__name__
    payload = _payload(native_event)
    lowered = event_name.lower()
    raw_type = _raw_item_type(payload).lower()
    classifier = f"{lowered} {raw_type}"
    if "approval" in classifier:
        event_type = "workflow"
        payload.setdefault("event", "approval")
    elif "guardrail" in classifier:
        event_type = "workflow"
        payload.setdefault("event", "guardrail")
    elif "handoff" in classifier or "agentupdated" in classifier:
        event_type = "workflow"
        payload.setdefault("event", "handoff")
    elif "tool" in classifier and ("output" in classifier or "result" in classifier):
        event_type = "tool_result"
    elif "function_call_output" in classifier:
        event_type = "tool_result"
    elif "tool" in classifier or "function_call" in classifier:
        event_type = "tool_call"
    elif "error" in classifier or "exception" in classifier:
        event_type = "error"
    else:
        event_type = "message"
    payload.setdefault("native_type", event_name)
    return AgentEvent(type=event_type, backend=backend, data=payload)


def _raw_item_type(payload: dict[str, Any]) -> str:
    raw_item = payload.get("raw_item")
    if isinstance(raw_item, dict):
        return str(raw_item.get("type", ""))
    return str(getattr(raw_item, "type", ""))


def _run_diagnostics(result: Any) -> dict[str, Any]:
    interruptions = list(getattr(result, "interruptions", []) or [])
    to_state = getattr(result, "to_state", None)
    return {
        "interruptions": _safe_summary(interruptions),
        "resumable": bool(interruptions and callable(to_state)),
        "state_type": _state_type(result) if interruptions else None,
        "last_agent": _safe_summary(getattr(result, "last_agent", None)),
        "last_response_id": getattr(result, "last_response_id", None),
        "raw_responses_count": len(getattr(result, "raw_responses", []) or []),
        "guardrails": {
            key: _safe_summary(value)
            for key, value in _guardrail_result_groups(result).items()
        },
    }


def _state_type(result: Any) -> str | None:
    to_state = getattr(result, "to_state", None)
    if not callable(to_state):
        return None
    try:
        return type(to_state()).__name__
    except Exception:  # pragma: no cover - defensive path for SDK/runtime failures.
        return None


def _guardrail_result_groups(result: Any) -> dict[str, list[Any]]:
    return {
        "input": list(getattr(result, "input_guardrail_results", []) or []),
        "output": list(getattr(result, "output_guardrail_results", []) or []),
        "tool_input": list(getattr(result, "tool_input_guardrail_results", []) or []),
        "tool_output": list(getattr(result, "tool_output_guardrail_results", []) or []),
    }


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _safe_summary(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): _safe_summary(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return type(value).__name__
