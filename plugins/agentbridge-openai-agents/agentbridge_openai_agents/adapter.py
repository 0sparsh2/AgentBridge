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
                "structured_output": "partial",
                "workflow.handoffs": "extension",
                "guardrails": "extension",
                "human_approval": "extension",
                "observability.tracing": "extension",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec name/instructions/model/tools to SDK Agent.",
                "agent.model": "Passes LiteLLM-style model strings through to the SDK; provider compatibility is SDK/model dependent.",
                "tools.sync": "Maps ToolSpec callables to OpenAI Agents function_tool wrappers.",
                "structured_output": "Passes AgentSpec.output_type to SDK Agent output_type when present.",
                "workflow.handoffs": "Planned through OpenAIAgentsExtension configuration.",
                "guardrails": "Planned through OpenAIAgentsExtension configuration.",
                "observability.tracing": "Planned through native trace/run metadata preservation.",
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
        if config.get("handoffs"):
            agent_kwargs["handoffs"] = config["handoffs"]
        if config.get("guardrails"):
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
        result = compiled_agent.sdk.Runner.run_sync(
            compiled_agent.native_agent,
            run_input.input,
            **_runner_kwargs(compiled_agent, run_input),
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
            metadata={"agent": compiled_agent.spec.name},
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
        del system_instructions, model_settings, output_schema, handoffs, tracing
        del previous_response_id, conversation_id, prompt

        items = input if isinstance(input, list) else [{"role": "user", "content": str(input)}]
        tool_output = _latest_tool_output(items)
        if tool_output is not None:
            return self._message_response(f"offline tool result: {tool_output}")
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
    if run_input.session_id:
        kwargs["conversation_id"] = run_input.session_id
    elif compiled.config.get("session_id"):
        kwargs["conversation_id"] = compiled.config["session_id"]
    if compiled.config.get("run_config"):
        kwargs["run_config"] = compiled.config["run_config"]
    elif _uses_offline_model(compiled) and hasattr(compiled.sdk, "RunConfig"):
        kwargs["run_config"] = compiled.sdk.RunConfig(tracing_disabled=True)
    return kwargs


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
    if "tool" in lowered and ("output" in lowered or "result" in lowered):
        event_type = "tool_result"
    elif "tool" in lowered:
        event_type = "tool_call"
    elif "error" in lowered or "exception" in lowered:
        event_type = "error"
    elif "handoff" in lowered or "agentupdated" in lowered:
        event_type = "workflow"
    else:
        event_type = "message"
    payload.setdefault("native_type", event_name)
    return AgentEvent(type=event_type, backend=backend, data=payload)


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}
