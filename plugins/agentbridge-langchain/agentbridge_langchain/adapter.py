"""AgentBridge adapter plugin for direct LangChain agents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.observability import callbacks_for_config, langsmith_context, observability_metadata
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
                "model.resilience": "extension",
                "tools.sync": "full",
                "tools.async": "partial",
                "tools.mcp": "extension",
                "structured_output": "full",
                "state.memory": "extension",
                "workflow.human_in_the_loop": "extension",
                "observability.tracing": "extension",
                "observability.diagnostics": "full",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec instructions to LangChain create_agent system_prompt.",
                "agent.model": "Normalizes provider/model to provider:model for LangChain provider parsing; provider support is environment dependent.",
                "model.resilience": "Applies native with_retry() before with_fallbacks() to supplied LangChain model objects.",
                "tools.sync": "Maps ToolSpec callables to LangChain StructuredTool instances.",
                "tools.async": "Preserves LangChain async runnables through native ainvoke/astream when exposed by the compiled agent.",
                "tools.mcp": "Passes native MCP adapter tools through LangChain create_agent without making langchain-mcp-adapters a core dependency.",
                "tools.retriever": "Wraps explicitly requested native retrievers as search tools while preserving the retriever implementation.",
                "structured_output": "Maps AgentSpec.output_type to LangChain response_format and validates native structured_response.",
                "state.memory": "Records memory/retriever hints and forwards native checkpointer/store objects when provided; portable memory semantics remain extension-level.",
                "observability.tracing": "Passes callbacks and metadata through native runtime config; provider-specific tracing remains extension-level.",
                "observability.diagnostics": "Summarizes messages, tool lifecycle events, structured responses, runtime config, and extension options.",
                "streaming.events": "Uses native stream/astream_events and normalizes message, tool, update, and custom event chunks.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledLangChainAgent:
        """Translate AgentSpec into a LangChain compiled agent graph."""

        create_agent, structured_tool = _load_langchain()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        native_tools = [_to_langchain_tool(structured_tool, tool) for tool in spec.tools]
        native_tools.extend(config.get("mcp_tools") or [])
        native_tools.extend(
            _to_langchain_retriever_tool(structured_tool, retriever, index)
            for index, retriever in enumerate(config.get("retriever_tools") or [])
        )
        model = config.get("model") or _model_for_spec(spec, config)
        model = _apply_model_resilience(model, config)
        agent_kwargs: dict[str, Any] = {
            "model": model,
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
        _copy_native_options(config.get("native_options"), agent_kwargs)

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
        with langsmith_context(compiled_agent.config):
            result = compiled_agent.native_agent.invoke(
                payload,
                config=runtime_config,
                **_context_kwargs(run_input),
            )
        return _normalized_result(compiled_agent, run_input, result, runtime_config)

    def batch(self, compiled: Any, run_inputs: Iterable[RunInput]) -> list[RunResult]:
        """Use LangChain's native batch path when available."""

        compiled_agent = _ensure_compiled(compiled)
        inputs = list(run_inputs)
        native_batch = getattr(compiled_agent.native_agent, "batch", None)
        if _uses_offline_model(compiled_agent) or not callable(native_batch):
            return super().batch(compiled_agent, inputs)
        payloads = [_input_payload(run_input) for run_input in inputs]
        configs = [_runtime_config(compiled_agent, run_input) for run_input in inputs]
        with langsmith_context(compiled_agent.config):
            results = native_batch(payloads, config=configs)
        return [
            _normalized_result(compiled_agent, run_input, result, config)
            for run_input, result, config in zip(inputs, results, configs, strict=True)
        ]

    async def arun(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Use LangChain's native async invoke path when available."""

        compiled_agent = _ensure_compiled(compiled)
        if not hasattr(compiled_agent.native_agent, "ainvoke"):
            return await super().arun(compiled_agent, run_input)
        runtime_config = _runtime_config(compiled_agent, run_input)
        with langsmith_context(compiled_agent.config):
            result = await compiled_agent.native_agent.ainvoke(
                _input_payload(run_input),
                config=runtime_config,
                **_context_kwargs(run_input),
            )
        return _normalized_result(compiled_agent, run_input, result, runtime_config)

    async def abatch(self, compiled: Any, run_inputs: Iterable[RunInput]) -> list[RunResult]:
        """Use LangChain's native async batch path when available."""

        compiled_agent = _ensure_compiled(compiled)
        inputs = list(run_inputs)
        native_batch = getattr(compiled_agent.native_agent, "abatch", None)
        if _uses_offline_model(compiled_agent) or not callable(native_batch):
            return await super().abatch(compiled_agent, inputs)
        payloads = [_input_payload(run_input) for run_input in inputs]
        configs = [_runtime_config(compiled_agent, run_input) for run_input in inputs]
        with langsmith_context(compiled_agent.config):
            results = await native_batch(payloads, config=configs)
        return [
            _normalized_result(compiled_agent, run_input, result, config)
            for run_input, result, config in zip(inputs, results, configs, strict=True)
        ]

    async def astream(self, compiled: Any, run_input: RunInput) -> AsyncIterator[AgentEvent]:
        """Consume LangChain's native async event stream when it is available."""

        compiled_agent = _ensure_compiled(compiled)
        if _uses_offline_model(compiled_agent):
            result = await self.arun(compiled_agent, run_input)
            for event in result.events:
                yield event
            return

        native_agent = compiled_agent.native_agent
        if not hasattr(native_agent, "astream") and not hasattr(native_agent, "astream_events"):
            async for event in super().astream(compiled_agent, run_input):
                yield event
            return

        yielded = False
        async for chunk in _native_astream(compiled_agent, run_input):
            yielded = True
            for event in _normalize_native_events(chunk, backend=self.backend_name):
                yield event
        if not yielded:
            result = await self.arun(compiled_agent, run_input)
            for event in result.events:
                yield event

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        if _uses_offline_model(compiled_agent):
            yield from self.run(compiled_agent, run_input).events
            return

        if not hasattr(compiled_agent.native_agent, "stream"):
            yield from self.run(compiled_agent, run_input).events
            return

        yielded = False
        for chunk in _native_stream(compiled_agent, run_input):
            yielded = True
            yield from _normalize_native_events(chunk, backend=self.backend_name)
        if not yielded:
            yield from self.run(compiled_agent, run_input).events


def _normalized_result(
    compiled_agent: CompiledLangChainAgent,
    run_input: RunInput,
    result: Any,
    runtime_config: dict[str, Any],
    events: list[AgentEvent] | None = None,
) -> RunResult:
    """Build the same normalized result shape for single and batch invocations."""

    output = _final_output(result)
    normalized_events = list(events or _events_from_result(result, backend="langchain"))
    normalized_events.extend(
        [
            AgentEvent(
                type="message",
                backend="langchain",
                data={"content": output, "agent": compiled_agent.spec.name},
            ),
            AgentEvent(type="complete", backend="langchain", data={"output": output}),
        ]
    )
    return RunResult(
        output=output,
        backend="langchain",
        events=normalized_events,
        metadata={
            "agent": compiled_agent.spec.name,
            "runtime_config": _safe_summary(runtime_config),
            "extension_config": _safe_summary(compiled_agent.config),
            "extension_summary": _extension_summary(compiled_agent.config),
            "native_agent_type": type(compiled_agent.native_agent).__name__,
            "run_diagnostics": _run_diagnostics(
                result,
                normalized_events,
                compiled_agent,
                runtime_config,
            ),
        },
        raw=result,
    )


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


def _copy_native_options(config: Any, agent_kwargs: dict[str, Any]) -> None:
    """Forward future LangChain options without allowing identity override."""

    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("LangChain native_options must be a dictionary.")
    reserved = {"model", "tools", "system_prompt", "response_format", "name"}
    conflicts = sorted(reserved.intersection(config))
    if conflicts:
        raise ValueError(
            "LangChain native_options cannot override AgentSpec-owned fields: "
            + ", ".join(conflicts)
        )
    agent_kwargs.update(config)


def _to_langchain_tool(structured_tool: Any, tool_spec: Any) -> Any:
    return structured_tool.from_function(
        func=tool_spec.handler,
        name=tool_spec.name,
        description=tool_spec.description,
    )


def _to_langchain_retriever_tool(structured_tool: Any, retriever: Any, index: int) -> Any:
    """Expose one native retriever as a query tool without replacing it."""

    raw_name = getattr(retriever, "name", None) or f"retriever_{index + 1}"
    name = "".join(character if character.isalnum() or character == "_" else "_" for character in raw_name)
    description = getattr(retriever, "description", None) or f"Search documents with {name}."

    def search(query: str) -> list[Any]:
        invoke = getattr(retriever, "invoke", None)
        if callable(invoke):
            documents = invoke(query)
        else:
            get_documents = getattr(retriever, "get_relevant_documents", None)
            if not callable(get_documents):
                raise TypeError(f"Retriever {name!r} must expose invoke() or get_relevant_documents().")
            documents = get_documents(query)
        return [_document_payload(document) for document in documents]

    return structured_tool.from_function(func=search, name=name, description=description)


def _document_payload(document: Any) -> Any:
    """Keep native document content and metadata while making results serializable."""

    if isinstance(document, dict):
        return document
    payload: dict[str, Any] = {}
    if hasattr(document, "page_content"):
        payload["page_content"] = document.page_content
    if hasattr(document, "metadata"):
        payload["metadata"] = document.metadata
    return payload or document


def _model_for_spec(spec: AgentSpec, config: dict[str, Any] | None = None) -> Any:
    if spec.model == "agentbridge/offline":
        model_base = import_module("langchain_core.language_models.chat_models").BaseChatModel

        class AgentBridgeOfflineModel(_AgentBridgeOfflineModelBase, model_base):
            pass

        return AgentBridgeOfflineModel()
    model_options = dict((config or {}).get("model_options") or {})
    if model_options:
        provider = (config or {}).get("model_provider") or _provider_from_model(spec.model)
        if provider in {"openai", "azure_openai", "openrouter", "nvidia_nim"}:
            try:
                module = import_module("langchain_openai")
                chat_openai = getattr(module, "AzureChatOpenAI" if provider == "azure_openai" else "ChatOpenAI")
            except ImportError as exc:
                raise ImportError(
                    "LangChain OpenAI-compatible model options require langchain-openai. "
                    "Install `agentbridge-langchain[openai]`."
                ) from exc
            model_options.setdefault("model", _model_name(spec.model))
            return chat_openai(**model_options)
        if provider == "ollama":
            try:
                chat_ollama = import_module("langchain_ollama").ChatOllama
            except ImportError as exc:
                raise ImportError(
                    "Ollama model options require langchain-ollama. "
                    "Install `agentbridge-langchain[ollama]`."
                ) from exc
            model_options.setdefault("model", _model_name(spec.model))
            return chat_ollama(**model_options)
        provider_model = _provider_model_factory(provider)
        if provider_model is not None:
            module_name, class_name, package_name = provider_model
            try:
                provider_class = getattr(import_module(module_name), class_name)
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    f"LangChain provider '{provider}' requires {package_name}. "
                    f"Install that provider integration before using model_options."
                ) from exc
            model_options.setdefault("model", _model_name(spec.model))
            return provider_class(**model_options)
        raise ValueError(f"Unsupported LangChain model_provider for model_options: {provider!r}")
    return _normalize_model(spec.model)


def _apply_model_resilience(model: Any, config: dict[str, Any]) -> Any:
    """Apply native LangChain retry/fallback wrappers without hiding failures."""

    retry_options = dict(config.get("model_retry") or {})
    if retry_options:
        with_retry = getattr(model, "with_retry", None)
        if not callable(with_retry):
            raise TypeError("LangChain model_retry requires a model exposing with_retry().")
        model = with_retry(**retry_options)

    fallbacks = list(config.get("model_fallbacks") or [])
    if fallbacks:
        if any(isinstance(fallback, str) for fallback in fallbacks):
            raise TypeError("LangChain model_fallbacks must contain native model objects, not strings.")
        with_fallbacks = getattr(model, "with_fallbacks", None)
        if not callable(with_fallbacks):
            raise TypeError("LangChain model_fallbacks requires a model exposing with_fallbacks().")
        model = with_fallbacks(fallbacks)
    return model


def _provider_from_model(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model.split(":", 1)[0]


def _model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _provider_model_factory(provider: str) -> tuple[str, str, str] | None:
    """Return a lazy LangChain provider class without importing provider packages eagerly."""

    factories = {
        "anthropic": ("langchain_anthropic", "ChatAnthropic", "langchain-anthropic"),
        "google": ("langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai"),
        "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai"),
        "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI", "langchain-google-vertexai"),
        "mistral": ("langchain_mistralai", "ChatMistralAI", "langchain-mistralai"),
        "groq": ("langchain_groq", "ChatGroq", "langchain-groq"),
        "cohere": ("langchain_cohere", "ChatCohere", "langchain-cohere"),
        "bedrock": ("langchain_aws", "ChatBedrock", "langchain-aws"),
        "fireworks": ("langchain_fireworks", "ChatFireworks", "langchain-fireworks"),
        "huggingface": ("langchain_huggingface", "ChatHuggingFace", "langchain-huggingface"),
        "xai": ("langchain_xai", "ChatXAI", "langchain-xai"),
    }
    return factories.get(provider)


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
            message = messages_module.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": getattr(tool, "name", "tool"),
                        "args": _arguments_for_tool(tool, messages),
                        "id": "agentbridge-offline-tool-call",
                    }
                ],
            )
        else:
            message = messages_module.AIMessage(
                content=f"offline response: {_last_user_text(messages)}"
            )

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


def _arguments_for_tool(tool: Any, messages: list[Any]) -> dict[str, Any]:
    args = getattr(tool, "args", {}) or {}
    fields = getattr(getattr(tool, "args_schema", None), "model_fields", {}) or {}
    if not args and fields:
        args = {
            name: getattr(field, "json_schema_extra", None) or {} for name, field in fields.items()
        }
    if not args:
        return {_first_tool_argument_name(tool): _last_user_text(messages)}

    required = _required_tool_arguments(tool)
    selected = required or list(args)
    return {name: _argument_value_for_schema(args.get(name, {}), messages) for name in selected}


def _required_tool_arguments(tool: Any) -> list[str]:
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, dict):
        return [str(item) for item in args_schema.get("required", [])]
    schema = getattr(args_schema, "model_json_schema", lambda: {})()
    if isinstance(schema, dict):
        return [str(item) for item in schema.get("required", [])]
    return []


def _argument_value_for_schema(schema: Any, messages: list[Any]) -> Any:
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump()
    if not isinstance(schema, dict):
        return _last_user_text(messages)
    if "default" in schema:
        return schema["default"]
    raw_type = schema.get("type", "string")
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
    return {
        "messages": [{"role": "user", "content": run_input.input}],
    }


def _context_kwargs(run_input: RunInput) -> dict[str, Any]:
    """Pass LangChain runtime context through its native invocation channel."""

    return {"context": run_input.context} if run_input.context else {}


def _runtime_config(compiled: CompiledLangChainAgent, run_input: RunInput) -> dict[str, Any]:
    config: dict[str, Any] = dict(compiled.config.get("runtime_config") or {})
    callbacks = callbacks_for_config(compiled.config)
    if callbacks:
        config["callbacks"] = callbacks
    metadata = observability_metadata(
        compiled.config,
        metadata=run_input.metadata,
        session_id=run_input.session_id,
    )
    observability = compiled.config.get("observability", {})
    if run_input.session_id:
        configurable = dict(config.get("configurable") or {})
        configurable["thread_id"] = run_input.session_id
        config["configurable"] = configurable
    tags = list(observability.get("tags", []))
    if tags:
        config["tags"] = tags
    run_name = observability.get("run_name")
    if run_name:
        config["run_name"] = run_name
    if metadata:
        config["metadata"] = metadata
    return config


def _native_stream(compiled: CompiledLangChainAgent, run_input: RunInput) -> Iterator[Any]:
    payload = _input_payload(run_input)
    runtime_config = _runtime_config(compiled, run_input)
    native_agent = compiled.native_agent

    with langsmith_context(compiled.config):
        if hasattr(native_agent, "stream_events"):
            try:
                yield from native_agent.stream_events(
                    payload,
                    config=runtime_config,
                    version="v3",
                    **_context_kwargs(run_input),
                )
                return
            except TypeError:
                pass

        try:
            yield from native_agent.stream(
                payload,
                config=runtime_config,
                stream_mode=["messages", "updates", "custom"],
                version="v2",
                **_context_kwargs(run_input),
            )
        except TypeError:
            yield from native_agent.stream(
                payload,
                config=runtime_config,
                **_context_kwargs(run_input),
            )


async def _native_astream(compiled: CompiledLangChainAgent, run_input: RunInput) -> AsyncIterator[Any]:
    payload = _input_payload(run_input)
    runtime_config = _runtime_config(compiled, run_input)
    native_agent = compiled.native_agent

    with langsmith_context(compiled.config):
        if hasattr(native_agent, "astream_events"):
            try:
                async for event in native_agent.astream_events(
                    payload,
                    config=runtime_config,
                    version="v3",
                    **_context_kwargs(run_input),
                ):
                    yield event
                return
            except TypeError:
                pass

        if hasattr(native_agent, "astream"):
            try:
                async for chunk in native_agent.astream(
                    payload,
                    config=runtime_config,
                    stream_mode=["messages", "updates", "custom"],
                    version="v2",
                    **_context_kwargs(run_input),
                ):
                    yield chunk
                return
            except TypeError:
                async for chunk in native_agent.astream(
                    payload,
                    config=runtime_config,
                    **_context_kwargs(run_input),
                ):
                    yield chunk


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
    summary = {
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
        "retriever_tools_count": len(config.get("retriever_tools") or []),
        "mcp_tools_count": len(config.get("mcp_tools") or []),
        "applied_native_options": applied_native_options,
        "native_options_count": len(config.get("native_options") or {}),
        "runtime_config": _safe_summary(config.get("runtime_config") or {}),
    }
    if config.get("observability"):
        summary["observability"] = _safe_summary(config["observability"])
    if config.get("agentcore"):
        summary["agentcore"] = _safe_summary(config["agentcore"])
    return summary


def _final_output(result: Any) -> Any:
    structured_response = _mapping_get(result, "structured_response")
    if structured_response is not None:
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
            str(item.get("text", item.get("content", item)))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content)


def _events_from_result(result: Any, *, backend: str) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    interrupts = _mapping_get(result, "__interrupt__")
    if interrupts:
        events.append(
            AgentEvent(
                type="workflow",
                backend=backend,
                data={"phase": "interrupted", "interrupts": _safe_summary(interrupts)},
            )
        )
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
        message_type = (
            message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
        )
        if message_type == "tool":
            events.append(
                AgentEvent(
                    type="tool_result",
                    backend=backend,
                    data=_payload(message),
                )
            )
    return events


def _run_diagnostics(
    result: Any,
    events: list[AgentEvent],
    compiled: CompiledLangChainAgent,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event.type] = event_counts.get(event.type, 0) + 1
    messages = list(_mapping_get(result, "messages") or [])
    interrupts = _mapping_get(result, "__interrupt__")
    return {
        "messages_count": len(messages),
        "tool_calls_count": sum(
            1 for message in messages for _tool_call in (_mapping_get(message, "tool_calls") or [])
        ),
        "tool_results_count": sum(
            1
            for message in messages
            if (
                message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
            )
            == "tool"
        ),
        "structured_response": _mapping_get(result, "structured_response") is not None,
        "interrupted": bool(interrupts),
        "interrupts_count": len(interrupts)
        if isinstance(interrupts, list | tuple)
        else int(bool(interrupts)),
        "runtime_config": _safe_summary(runtime_config),
        "extension_summary": _extension_summary(compiled.config),
        "native_agent_type": type(compiled.native_agent).__name__,
        "event_counts": event_counts,
    }


def _normalize_native_events(native_event: Any, *, backend: str) -> Iterator[AgentEvent]:
    payload = _payload(native_event)
    event_type = str(payload.get("type", "")).lower()

    if payload.get("__interrupt__"):
        yield AgentEvent(
            type="workflow",
            backend=backend,
            data={"phase": "interrupted", "interrupts": _safe_summary(payload["__interrupt__"])},
        )
        return

    if event_type == "messages":
        yield from _events_from_message_stream_data(payload.get("data"), backend=backend)
        return
    if event_type == "updates":
        yield from _events_from_update_data(payload.get("data"), backend=backend)
        return
    if event_type == "custom":
        payload.setdefault("native_type", type(native_event).__name__)
        yield AgentEvent(type="workflow", backend=backend, data=payload)
        return

    if _looks_like_update_payload(payload):
        yield from _events_from_update_data(payload, backend=backend)
        return

    messages = payload.get("messages")
    if messages:
        yielded = False
        for event in _events_from_result({"messages": messages}, backend=backend):
            yielded = True
            yield event
        if not yielded:
            for message in messages:
                yield AgentEvent(
                    type="message",
                    backend=backend,
                    data={
                        "content": _message_content(message),
                        "native_type": type(message).__name__,
                    },
                )
        return

    yield _normalize_native_event(native_event, backend=backend)


def _events_from_message_stream_data(data: Any, *, backend: str) -> Iterator[AgentEvent]:
    token, metadata = _split_message_stream_data(data)
    token_payload = _payload(token)
    metadata_payload = _safe_summary(metadata or {})

    for tool_call in _message_tool_call_chunks(token):
        yield AgentEvent(
            type="tool_call",
            backend=backend,
            data={
                **_payload(tool_call),
                "delta": True,
                "metadata": metadata_payload,
                "native_type": type(token).__name__,
            },
        )

    for tool_call in _message_tool_calls(token):
        yield AgentEvent(
            type="tool_call",
            backend=backend,
            data={
                **_payload(tool_call),
                "delta": False,
                "metadata": metadata_payload,
                "native_type": type(token).__name__,
            },
        )

    text = _message_text_delta(token)
    if text:
        yield AgentEvent(
            type="message",
            backend=backend,
            data={
                "content": text,
                "metadata": metadata_payload,
                "native_type": type(token).__name__,
            },
        )
    elif not _message_tool_call_chunks(token) and not _message_tool_calls(token):
        token_payload.setdefault("metadata", metadata_payload)
        token_payload.setdefault("native_type", type(token).__name__)
        yield AgentEvent(type="message", backend=backend, data=token_payload)


def _events_from_update_data(data: Any, *, backend: str) -> Iterator[AgentEvent]:
    if not isinstance(data, dict):
        yield AgentEvent(
            type="workflow",
            backend=backend,
            data={"value": data, "native_type": type(data).__name__},
        )
        return

    yielded = False
    for source, update in data.items():
        messages = _mapping_get(update, "messages")
        if messages:
            for event in _events_from_result({"messages": messages}, backend=backend):
                event.metadata["source"] = str(source)
                yielded = True
                yield event
            continue
        yield AgentEvent(
            type="workflow",
            backend=backend,
            data={
                "source": str(source),
                "update": _safe_summary(update),
                "native_type": type(update).__name__,
            },
        )
        yielded = True
    if not yielded:
        yield AgentEvent(
            type="workflow",
            backend=backend,
            data={"update": _safe_summary(data), "native_type": type(data).__name__},
        )


def _split_message_stream_data(data: Any) -> tuple[Any, Any]:
    if isinstance(data, list | tuple) and data:
        token = data[0]
        metadata = data[1] if len(data) > 1 else {}
        return token, metadata
    return data, {}


def _message_tool_call_chunks(message: Any) -> list[Any]:
    chunks = _mapping_get(message, "tool_call_chunks")
    return list(chunks or [])


def _message_tool_calls(message: Any) -> list[Any]:
    calls = _mapping_get(message, "tool_calls")
    return list(calls or [])


def _message_text_delta(message: Any) -> str:
    if isinstance(message, dict) and "text" in message:
        return str(message.get("text") or "")
    if not isinstance(message, dict) and hasattr(message, "text"):
        return str(getattr(message, "text") or "")
    text = _mapping_get(message, "text")
    if text:
        return str(text)
    return _message_content(message)


def _looks_like_update_payload(payload: dict[str, Any]) -> bool:
    if "type" in payload:
        return False
    return any(isinstance(value, dict) and "messages" in value for value in payload.values())


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
        return {
            str(key): "[redacted]" if _is_secret_key(str(key)) else _safe_summary(item)
            for key, item in value.items()
        }
    if hasattr(value, "__dict__"):
        return {
            str(key): _safe_summary(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return type(value).__name__


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(token in normalized for token in ("api_key", "token", "secret", "password"))
