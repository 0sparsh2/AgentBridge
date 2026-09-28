"""AgentBridge adapter for LangChain Deep Agents."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.observability import callbacks_for_config, langsmith_context
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CompiledDeepAgent:
    spec: AgentSpec
    native_agent: Any
    config: dict[str, Any]


class Adapter(BackendAdapter):
    """Translate AgentSpec into a native Deep Agents compiled graph."""

    backend_name = "deepagents"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "structured_output": "extension",
                "state.memory": "extension",
                "state.subagents": "extension",
                "filesystem.backend": "native_only",
                "human_approval": "extension",
                "streaming.events": "partial",
                "observability.tracing": "extension",
                "observability.raw": "full",
            },
            notes={
                "agent.model": "Provider:model strings and initialized native models are passed to create_deep_agent.",
                "state.memory": "Deep Agents memory/skills paths are forwarded unchanged.",
                "filesystem.backend": "Backend and permission objects remain native escape hatches.",
                "observability.tracing": "Callbacks and metadata use the shared AgentBridge observability config.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledDeepAgent:
        create_deep_agent = _load_create_deep_agent()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        kwargs: dict[str, Any] = {
            "model": config.get("model") or _model_for_spec(spec),
            "tools": [tool.handler for tool in spec.tools],
            "system_prompt": config.get("system_prompt") or spec.instructions,
            "name": config.get("name") or spec.name,
        }
        for option in (
            "middleware",
            "subagents",
            "skills",
            "memory",
            "permissions",
            "backend",
            "interrupt_on",
            "response_format",
            "state_schema",
            "context_schema",
            "checkpointer",
            "store",
            "debug",
            "cache",
        ):
            if config.get(option) is not None:
                kwargs[option] = config[option]
        _copy_native_options(config.get("native_options"), kwargs)
        return CompiledDeepAgent(spec=spec, native_agent=create_deep_agent(**kwargs), config=config)

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        compiled_agent = _ensure_compiled(compiled)
        runtime_config = _runtime_config(compiled_agent, run_input)
        with langsmith_context(compiled_agent.config):
            result = compiled_agent.native_agent.invoke(
                {"messages": [{"role": "user", "content": run_input.input}]},
                config=runtime_config,
            )
        output = _final_output(result)
        events = [
            AgentEvent(type="message", backend=self.backend_name, data={"content": output}),
            AgentEvent(type="complete", backend=self.backend_name, data={"output": output}),
        ]
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "agent": compiled_agent.spec.name,
                "runtime_config": _safe_summary(runtime_config),
                "native_agent_type": type(compiled_agent.native_agent).__name__,
                "native_options_count": len(compiled_agent.config.get("native_options") or {}),
            },
            raw=result,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        if not hasattr(compiled_agent.native_agent, "stream"):
            yield from self.run(compiled_agent, run_input).events
            return
        payload = {"messages": [{"role": "user", "content": run_input.input}]}
        runtime_config = _runtime_config(compiled_agent, run_input)
        with langsmith_context(compiled_agent.config):
            try:
                chunks = compiled_agent.native_agent.stream(payload, config=runtime_config, stream_mode="messages")
            except TypeError:
                chunks = compiled_agent.native_agent.stream(payload, config=runtime_config)
            yielded = False
            for chunk in chunks:
                yielded = True
                yield AgentEvent(
                    type="workflow",
                    backend=self.backend_name,
                    data={"chunk": _safe_summary(chunk)},
                )
        if not yielded:
            yield from self.run(compiled_agent, run_input).events


def _load_create_deep_agent() -> Any:
    try:
        return import_module("deepagents").create_deep_agent
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Deep Agents is not installed. Install `agentbridge-deepagents`."
        ) from exc


def _model_for_spec(spec: AgentSpec) -> Any:
    if spec.model == "agentbridge/offline":
        raise ValueError("Deep Agents requires a native model or provider:model string; offline is not supported.")
    if ":" in spec.model:
        return spec.model
    if "/" in spec.model:
        provider, model = spec.model.split("/", 1)
        return f"{provider}:{model}"
    return spec.model


def _copy_native_options(config: Any, kwargs: dict[str, Any]) -> None:
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("Deep Agents native_options must be a dictionary.")
    reserved = {"model", "tools", "system_prompt", "name"}
    conflicts = sorted(reserved.intersection(config))
    if conflicts:
        raise ValueError("Deep Agents native_options cannot override: " + ", ".join(conflicts))
    kwargs.update(config)


def _runtime_config(compiled: CompiledDeepAgent, run_input: RunInput) -> dict[str, Any]:
    config: dict[str, Any] = {}
    callbacks = callbacks_for_config(compiled.config)
    if callbacks:
        config["callbacks"] = callbacks
    metadata = dict(compiled.config.get("metadata", {}))
    metadata.update(run_input.metadata)
    if run_input.session_id:
        metadata["session_id"] = run_input.session_id
        config["configurable"] = {"thread_id": run_input.session_id}
    if metadata:
        config["metadata"] = metadata
    tags = compiled.config.get("observability", {}).get("tags") or []
    if tags:
        config["tags"] = list(tags)
    return config


def _ensure_compiled(value: Any) -> CompiledDeepAgent:
    if not isinstance(value, CompiledDeepAgent):
        raise TypeError("Deep Agents adapter expected CompiledDeepAgent from compile().")
    return value


def _final_output(result: Any) -> Any:
    if isinstance(result, dict):
        if result.get("structured_response") is not None:
            return result["structured_response"]
        messages = result.get("messages") or []
        if messages:
            message = messages[-1]
            return message.get("content", message) if isinstance(message, dict) else getattr(message, "content", message)
        return result.get("output", result)
    return getattr(result, "output", result)


def _safe_summary(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_summary(item) for item in value]
    return type(value).__name__

