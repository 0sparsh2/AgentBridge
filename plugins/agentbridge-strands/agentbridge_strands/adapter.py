"""AgentBridge adapter plugin for Strands Agents."""

from __future__ import annotations

import asyncio
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
                "structured_output": "partial",
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
                "tools.mcp": "Planned through StrandsExtension mcp_clients.",
                "structured_output": "Passes AgentSpec.output_type to Strands structured_output_model when present.",
                "observability.tracing": "Planned through trace_attributes and native run metadata.",
                "streaming.events": "Uses Strands stream_async when available and normalizes events best-effort.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledStrandsAgent:
        """Translate AgentSpec into a Strands Agent."""

        sdk = _load_strands_sdk()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        native_tools = [_to_strands_tool(sdk, tool) for tool in spec.tools]
        agent_kwargs: dict[str, Any] = {
            "name": spec.name,
            "model": spec.model,
            "system_prompt": spec.instructions,
            "tools": native_tools,
        }
        if spec.output_type is not None:
            agent_kwargs["structured_output_model"] = spec.output_type
        if config.get("conversation_manager"):
            agent_kwargs["conversation_manager"] = config["conversation_manager"]
        if config.get("trace_attributes"):
            agent_kwargs["trace_attributes"] = config["trace_attributes"]
        if config.get("hooks"):
            agent_kwargs["hooks"] = config["hooks"]
        if config.get("metadata"):
            agent_kwargs["state"] = {"metadata": config["metadata"]}

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
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=[
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
            ],
            usage=_usage_from_result(result),
            metadata={
                "agent": compiled_agent.spec.name,
                "interrupts": _safe_repr(getattr(result, "interrupts", None)),
                "checkpoint": _safe_repr(getattr(result, "checkpoint", None)),
            },
            raw=result,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
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


def _final_output(result: Any) -> Any:
    structured_output = getattr(result, "structured_output", None)
    if structured_output is not None:
        if hasattr(structured_output, "model_dump"):
            return structured_output.model_dump()
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
