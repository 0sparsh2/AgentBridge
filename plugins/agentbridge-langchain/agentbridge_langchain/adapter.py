"""AgentBridge adapter plugin for direct LangChain agents."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CompiledLangChainAgent:
    """Compiled LangChain graph/runnable plus the source AgentBridge spec."""

    spec: AgentSpec
    native_agent: Any
    native_tools: list[Any]
    config: dict[str, Any]


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "langchain"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "tools.async": "partial",
                "state.memory": "extension",
                "observability.tracing": "extension",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec instructions to LangChain create_agent system_prompt.",
                "agent.model": "Normalizes provider/model to provider:model for LangChain provider parsing; provider support is environment dependent.",
                "tools.sync": "Maps ToolSpec callables to LangChain StructuredTool instances.",
                "tools.async": "LangChain supports async runnables, but this adapter currently exposes synchronous run and best-effort stream.",
                "state.memory": "Planned through LangChainExtension memory configuration.",
                "observability.tracing": "Planned through callbacks and tracing metadata.",
                "streaming.events": "Uses native stream() when available and normalizes event chunks best-effort.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledLangChainAgent:
        """Translate AgentSpec into a LangChain compiled agent graph."""

        create_agent, structured_tool = _load_langchain()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        native_tools = [_to_langchain_tool(structured_tool, tool) for tool in spec.tools]
        agent_kwargs: dict[str, Any] = {
            "model": config.get("model") or _normalize_model(spec.model),
            "tools": native_tools,
            "system_prompt": config.get("prompt_template") or spec.instructions,
            "name": spec.name,
        }
        if spec.output_type is not None:
            agent_kwargs["response_format"] = spec.output_type
        if config.get("middleware"):
            agent_kwargs["middleware"] = config["middleware"]
        if config.get("checkpointer"):
            agent_kwargs["checkpointer"] = config["checkpointer"]
        if config.get("store"):
            agent_kwargs["store"] = config["store"]
        if config.get("debug") is not None:
            agent_kwargs["debug"] = bool(config["debug"])

        native_agent = create_agent(**agent_kwargs)
        return CompiledLangChainAgent(
            spec=spec,
            native_agent=native_agent,
            native_tools=native_tools,
            config=config,
        )

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled LangChain agent and return a normalized result."""

        compiled_agent = _ensure_compiled(compiled)
        payload = _input_payload(run_input)
        result = compiled_agent.native_agent.invoke(
            payload,
            config=_runtime_config(compiled_agent, run_input),
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
                    },
                ),
                AgentEvent(
                    type="complete",
                    backend=self.backend_name,
                    data={"output": output},
                ),
            ],
            metadata={"agent": compiled_agent.spec.name},
            raw=result,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        if not hasattr(compiled_agent.native_agent, "stream"):
            yield from self.run(compiled_agent, run_input).events
            return

        yielded = False
        for chunk in compiled_agent.native_agent.stream(
            _input_payload(run_input),
            config=_runtime_config(compiled_agent, run_input),
        ):
            yielded = True
            yield _normalize_native_event(chunk, backend=self.backend_name)
        if not yielded:
            yield from self.run(compiled_agent, run_input).events


def _load_langchain() -> tuple[Any, Any]:
    try:
        create_agent = import_module("langchain.agents").create_agent
        structured_tool = import_module("langchain_core.tools").StructuredTool
    except ImportError as exc:  # pragma: no cover - message assertion path.
        raise ImportError(
            "LangChain is not installed. Install this plugin with `langchain>=1.4,<2`."
        ) from exc
    return create_agent, structured_tool


def _to_langchain_tool(structured_tool: Any, tool_spec: Any) -> Any:
    return structured_tool.from_function(
        func=tool_spec.handler,
        name=tool_spec.name,
        description=tool_spec.description,
    )


def _ensure_compiled(compiled: Any) -> CompiledLangChainAgent:
    if not isinstance(compiled, CompiledLangChainAgent):
        raise TypeError("LangChain adapter expected CompiledLangChainAgent from compile().")
    return compiled


def _normalize_model(model: str) -> str:
    if ":" in model:
        return model
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return f"{provider}:{model_name}"
    return model


def _input_payload(run_input: RunInput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": run_input.input}],
    }
    if run_input.context:
        payload["context"] = run_input.context
    return payload


def _runtime_config(compiled: CompiledLangChainAgent, run_input: RunInput) -> dict[str, Any]:
    config: dict[str, Any] = {}
    callbacks = compiled.config.get("callbacks")
    if callbacks:
        config["callbacks"] = callbacks
    metadata = dict(compiled.config.get("metadata", {}))
    metadata.update(run_input.metadata)
    if run_input.session_id:
        metadata["session_id"] = run_input.session_id
        config["configurable"] = {"thread_id": run_input.session_id}
    if metadata:
        config["metadata"] = metadata
    return config


def _final_output(result: Any) -> Any:
    structured_response = _mapping_get(result, "structured_response")
    if structured_response is not None:
        if hasattr(structured_response, "model_dump"):
            return structured_response.model_dump()
        return structured_response
    messages = _mapping_get(result, "messages")
    if messages:
        return _message_content(messages[-1])
    output = _mapping_get(result, "output")
    if output is not None:
        return output
    return result


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", message)
    else:
        content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", item.get("content", item))) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _normalize_native_event(native_event: Any, *, backend: str) -> AgentEvent:
    payload = _payload(native_event)
    keys = {str(key).lower() for key in payload}
    if any("tool" in key and ("result" in key or "output" in key) for key in keys):
        event_type = "tool_result"
    elif any("tool" in key for key in keys):
        event_type = "tool_call"
    elif any("error" in key or "exception" in key for key in keys):
        event_type = "error"
    else:
        event_type = "message"
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
