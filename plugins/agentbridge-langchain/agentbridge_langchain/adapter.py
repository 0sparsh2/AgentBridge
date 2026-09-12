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
    offline: bool = False


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
                "state.memory": "Records memory/retriever hints and forwards native checkpointer/store objects when provided; portable memory semantics remain extension-level.",
                "observability.tracing": "Passes callbacks and metadata through native runtime config; provider-specific tracing remains extension-level.",
                "streaming.events": "Uses native stream() when available and normalizes event chunks best-effort.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledLangChainAgent:
        """Translate AgentSpec into a LangChain compiled agent graph."""

        create_agent, structured_tool = _load_langchain()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        native_tools = [_to_langchain_tool(structured_tool, tool) for tool in spec.tools]
        agent_kwargs: dict[str, Any] = {
            "model": config.get("model") or _model_for_spec(spec),
            "tools": native_tools,
            "system_prompt": config.get("prompt_template") or spec.instructions,
            "name": spec.name,
        }
        if spec.output_type is not None:
            agent_kwargs["response_format"] = spec.output_type
        if config.get("middleware"):
            agent_kwargs["middleware"] = config["middleware"]
        _copy_native_create_agent_options(
            config,
            agent_kwargs,
            (
                "checkpointer",
                "store",
                "interrupt_before",
                "interrupt_after",
                "cache",
                "state_schema",
                "context_schema",
                "transformers",
                "debug",
            ),
        )

        native_agent = create_agent(**agent_kwargs)
        return CompiledLangChainAgent(
            spec=spec,
            native_agent=native_agent,
            native_tools=native_tools,
            config=config,
            offline=spec.model == "agentbridge/offline",
        )

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled LangChain agent and return a normalized result."""

        compiled_agent = _ensure_compiled(compiled)
        payload = _input_payload(run_input)
        runtime_config = _runtime_config(compiled_agent, run_input)
        result = compiled_agent.native_agent.invoke(
            payload,
            config=runtime_config,
        )
        output = _final_output(result)
        events = _events_from_result(result, backend=self.backend_name)
        events.extend(
            [
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
            ]
        )
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "agent": compiled_agent.spec.name,
                "runtime_config": _safe_summary(runtime_config),
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


def _copy_native_create_agent_options(
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
        if option_name == "debug":
            agent_kwargs[option_name] = bool(value)
        else:
            agent_kwargs[option_name] = value


def _to_langchain_tool(structured_tool: Any, tool_spec: Any) -> Any:
    return structured_tool.from_function(
        func=tool_spec.handler,
        name=tool_spec.name,
        description=tool_spec.description,
    )


def _model_for_spec(spec: AgentSpec) -> Any:
    if spec.model == "agentbridge/offline":
        model_base = import_module("langchain_core.language_models.chat_models").BaseChatModel

        class AgentBridgeOfflineModel(_AgentBridgeOfflineModelBase, model_base):
            pass

        return AgentBridgeOfflineModel()
    return _normalize_model(spec.model)


class _AgentBridgeOfflineModelBase:
    """No-network LangChain chat model used by conformance tests."""

    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "agentbridge-offline"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        del tool_choice, kwargs
        return self.model_copy(update={"bound_tools": list(tools or [])})

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        del stop, run_manager, kwargs
        messages_module = import_module("langchain_core.messages")
        outputs_module = import_module("langchain_core.outputs")

        tool_result = _latest_tool_message_content(messages)
        if tool_result is not None:
            message = messages_module.AIMessage(content=f"offline tool result: {tool_result}")
        elif self.bound_tools:
            tool = self.bound_tools[0]
            argument_name = _first_tool_argument_name(tool)
            message = messages_module.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": getattr(tool, "name", "tool"),
                        "args": {argument_name: _last_user_text(messages)},
                        "id": "agentbridge-offline-tool-call",
                    }
                ],
            )
        else:
            message = messages_module.AIMessage(content=f"offline response: {_last_user_text(messages)}")

        return outputs_module.ChatResult(
            generations=[outputs_module.ChatGeneration(message=message)]
        )


def _uses_offline_model(compiled: CompiledLangChainAgent) -> bool:
    return compiled.offline


def _first_tool_argument_name(tool: Any) -> str:
    args = getattr(tool, "args", {}) or {}
    if args:
        return str(next(iter(args)))
    fields = getattr(getattr(tool, "args_schema", None), "model_fields", {}) or {}
    if fields:
        return str(next(iter(fields)))
    return "input"


def _latest_tool_message_content(messages: list[Any]) -> Any | None:
    for message in reversed(messages):
        if getattr(message, "type", None) == "tool":
            return getattr(message, "content", None)
    return None


def _last_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) in {"human", "user"}:
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


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


def _extension_summary(config: dict[str, Any]) -> dict[str, Any]:
    native_agent_options = [
        "checkpointer",
        "store",
        "interrupt_before",
        "interrupt_after",
        "cache",
        "state_schema",
        "context_schema",
        "transformers",
        "debug",
    ]
    applied_native_options = [
        option_name
        for option_name in native_agent_options
        if option_name in config and config[option_name] is not None
    ]
    return {
        "agent_type": config.get("agent_type"),
        "prompt_template": bool(config.get("prompt_template")),
        "middleware_count": len(config.get("middleware") or []),
        "callbacks_count": len(config.get("callbacks") or []),
        "memory": {
            "requested": config.get("memory"),
            "native_checkpointer": "checkpointer" in applied_native_options,
        },
        "retrievers": {
            "requested": _safe_summary(config.get("retrievers") or []),
            "native_store": "store" in applied_native_options,
        },
        "applied_native_options": applied_native_options,
    }


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


def _events_from_result(result: Any, *, backend: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for message in _mapping_get(result, "messages") or []:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls", tool_calls)
        for tool_call in tool_calls or []:
            events.append(
                AgentEvent(
                    type="tool_call",
                    backend=backend,
                    data=_payload(tool_call),
                )
            )
        message_type = message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
        if message_type == "tool":
            events.append(
                AgentEvent(
                    type="tool_result",
                    backend=backend,
                    data=_payload(message),
                )
            )
    return events


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


def _safe_summary(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    return type(value).__name__
