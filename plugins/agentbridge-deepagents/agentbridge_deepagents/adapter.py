"""Adapter for LangChain's Deep Agents harness."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.errors import MissingDependencyError
from agentbridge.observability import callbacks_for_config, langsmith_context, observability_metadata
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CompiledDeepAgent:
    spec: AgentSpec
    native_agent: Any
    config: dict[str, Any]


class Adapter(BackendAdapter):
    """Translate AgentSpec into ``deepagents.create_deep_agent``."""

    backend_name = "deepagents"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "structured_output": "extension",
                "tools.mcp": "extension",
                "workflow.subagents": "extension",
                "workflow.skills": "extension",
                "state.filesystem": "extension",
                "state.memory": "extension",
                "state.checkpointing": "extension",
                "protocols.ag_ui": "extension",
                "protocols.a2a": "extension",
                "sandbox.execution": "native_only",
                "human_approval": "extension",
                "streaming.events": "partial",
                "observability.tracing": "extension",
                "observability.raw": "full",
                "observability.diagnostics": "full",
            },
            notes={
                "workflow.subagents": "Pass native SubAgent, CompiledSubAgent, or AsyncSubAgent objects through subagents.",
                "workflow.skills": "Pass skill source paths through skills; backend root and file loading remain native.",
                "state.filesystem": "Pass StateBackend, FilesystemBackend, or SandboxBackend through backend.",
                "protocols.ag_ui": "Protocol labels and native transport options remain extension metadata.",
                "protocols.a2a": "Protocol labels and native transport options remain extension metadata.",
                "sandbox.execution": "Sandbox and provider objects remain native escape hatches.",
                "human_approval": "Pass interrupt_on and permissions to native Deep Agents middleware.",
                "observability.tracing": "Callbacks and metadata use shared AgentBridge observability configuration.",
                "observability.raw": "Preserves the native Deep Agent object and all native options.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledDeepAgent:
        create_deep_agent = _load_create_deep_agent()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        structured_tool = _load_structured_tool()
        tools = [
            structured_tool.from_function(
                func=tool.handler,
                name=tool.name,
                description=tool.description,
            )
            for tool in spec.tools
        ]
        kwargs: dict[str, Any] = {
            "name": config.get("name") or spec.name,
            "model": config.get("model") or _normalize_model(spec.model),
            "tools": tools,
            "system_prompt": config.get("system_prompt") or spec.instructions,
        }
        for option in (
            "middleware",
            "subagents",
            "skills",
            "memory",
            "permissions",
            "backend",
            "checkpointer",
            "store",
            "interrupt_on",
            "response_format",
            "state_schema",
            "context_schema",
            "general_purpose_subagent",
            "debug",
            "cache",
        ):
            if config.get(option) is not None:
                kwargs[option] = config[option]
        if spec.output_type is not None:
            kwargs["response_format"] = spec.output_type
        _copy_native_options(config.get("native_options"), kwargs)
        return CompiledDeepAgent(spec=spec, native_agent=create_deep_agent(**kwargs), config=config)

    def run(self, compiled: CompiledDeepAgent, run_input: RunInput) -> RunResult:
        compiled = _ensure_compiled(compiled)
        runtime_config = _runtime_config(compiled, run_input)
        kwargs = {"config": runtime_config} if runtime_config else {}
        if run_input.context:
            kwargs["context"] = run_input.context
        with langsmith_context(compiled.config):
            raw = compiled.native_agent.invoke(
                {"messages": [{"role": "user", "content": run_input.input}]},
                **kwargs,
            )
        output = _final_output(raw)
        events = [
            AgentEvent(type="message", backend=self.backend_name, data={"content": output}),
            AgentEvent(type="complete", backend=self.backend_name, data={"output": output}),
        ]
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "agent": compiled.spec.name,
                "runtime_config": runtime_config,
                "native_agent_type": type(compiled.native_agent).__name__,
                "deep_agent_options": _safe_summary(compiled.config),
                "native_options_count": len(compiled.config.get("native_options") or {}),
                "protocols": list(compiled.config.get("protocols") or []),
                "sandbox": _safe_summary(compiled.config.get("sandbox")),
            },
            raw=raw,
        )

    def stream(self, compiled: CompiledDeepAgent, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled = _ensure_compiled(compiled)
        native_stream = getattr(compiled.native_agent, "stream", None)
        if not callable(native_stream):
            yield from self.run(compiled, run_input).events
            return
        kwargs = _runtime_config(compiled, run_input)
        stream_kwargs = {"config": kwargs} if kwargs else {}
        if run_input.context:
            stream_kwargs["context"] = run_input.context
        yielded = False
        last_output: Any = None
        with langsmith_context(compiled.config):
            try:
                chunks = native_stream(
                    {"messages": [{"role": "user", "content": run_input.input}]},
                    **stream_kwargs,
                    stream_mode="messages",
                )
            except TypeError:
                chunks = native_stream(
                    {"messages": [{"role": "user", "content": run_input.input}]},
                    **stream_kwargs,
                )
            for chunk in chunks:
                yielded = True
                event, output = _event_from_chunk(chunk)
                if output is not None:
                    last_output = output
                if event is not None:
                    yield event
        if not yielded:
            yield from self.run(compiled, run_input).events
            return
        yield AgentEvent(type="complete", backend=self.backend_name, data={"output": last_output})


def _load_create_deep_agent() -> Any:
    try:
        return import_module("deepagents").create_deep_agent
    except (ImportError, AttributeError) as exc:
        raise MissingDependencyError("deepagents", "deepagents", "runtime") from exc


def _load_structured_tool() -> Any:
    try:
        return import_module("langchain_core.tools").StructuredTool
    except ImportError as exc:
        raise MissingDependencyError("deepagents", "langchain-core", "runtime") from exc


def _copy_native_options(config: Any, kwargs: dict[str, Any]) -> None:
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("Deep Agents native_options must be a dictionary.")
    reserved = {"name", "model", "tools", "system_prompt", "response_format"}
    conflicts = sorted(reserved.intersection(config))
    if conflicts:
        raise ValueError("Deep Agents native_options cannot override: " + ", ".join(conflicts))
    kwargs.update(config)


def _normalize_model(model: str) -> str:
    if model == "agentbridge/offline":
        raise ValueError("Deep Agents requires a native model or provider:model string; offline is not supported.")
    if ":" in model:
        return model
    if "/" in model:
        provider, name = model.split("/", 1)
        return f"{provider}:{name}"
    return model


def _runtime_config(compiled: CompiledDeepAgent, run_input: RunInput) -> dict[str, Any]:
    config: dict[str, Any] = {}
    callbacks = callbacks_for_config(compiled.config)
    if callbacks:
        config["callbacks"] = callbacks
    metadata = observability_metadata(
        {"metadata": compiled.spec.metadata, "observability": compiled.config.get("observability", {})},
        metadata=run_input.metadata,
        session_id=run_input.session_id,
    )
    if metadata:
        config["metadata"] = metadata
    if run_input.session_id:
        config["configurable"] = {"thread_id": run_input.session_id}
    tags = compiled.config.get("observability", {}).get("tags") or []
    if tags:
        config["tags"] = list(tags)
    return config


def _ensure_compiled(value: Any) -> CompiledDeepAgent:
    if not isinstance(value, CompiledDeepAgent):
        raise TypeError("Deep Agents adapter expected CompiledDeepAgent from compile().")
    return value


def _final_output(raw: Any) -> Any:
    if isinstance(raw, dict):
        if raw.get("structured_response") is not None:
            return raw["structured_response"]
        messages = raw.get("messages") or []
        if messages:
            message = messages[-1]
            return message.get("content") if isinstance(message, dict) else getattr(message, "content", message)
        return raw.get("output", raw)
    return raw


def _event_from_chunk(chunk: Any) -> tuple[AgentEvent | None, Any | None]:
    if not isinstance(chunk, dict):
        return AgentEvent(type="workflow", backend="deepagents", data={"value": chunk}), None
    output = _final_output(chunk) if "messages" in chunk or "output" in chunk else None
    return AgentEvent(type="workflow", backend="deepagents", data={"update": _safe_summary(chunk)}), output


def _safe_summary(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items() if str(key) not in {"handler", "api_key"}}
    if isinstance(value, (list, tuple)):
        return [_safe_summary(item) for item in value]
    return {"type": type(value).__name__}
