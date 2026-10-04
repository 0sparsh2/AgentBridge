from __future__ import annotations

from contextlib import contextmanager
import sys
from types import SimpleNamespace

from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge_langchain.adapter import Adapter, CompiledLangChainAgent
from agentbridge_langchain.adapter import _normalize_native_events


class FakeStructuredTool:
    @staticmethod
    def from_function(**kwargs):
        return SimpleNamespace(**kwargs)


class FakeNativeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_config = None

    def invoke(self, payload, config=None):
        self.last_config = config
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"native langchain: {payload['messages'][0]['content']}",
                }
            ]
        }

    def stream(self, payload, config=None):
        yield {"messages": [{"content": "stream chunk"}]}


class FakeEventStreamAgent(FakeNativeAgent):
    def stream_events(self, payload, config=None, version=None):
        del payload, config
        assert version == "v3"
        yield {
            "type": "messages",
            "data": (
                SimpleNamespace(
                    text="",
                    tool_call_chunks=[
                        {
                            "name": "lookup_order",
                            "args": '{"order_id"',
                            "id": "call-1",
                            "index": 0,
                        }
                    ],
                ),
                {"lc_agent_name": "support_agent"},
            ),
        }
        yield {
            "type": "updates",
            "data": {
                "model": {
                    "messages": [
                        SimpleNamespace(
                            type="ai",
                            content="",
                            tool_calls=[
                                {
                                    "name": "lookup_order",
                                    "args": {"order_id": "A123"},
                                    "id": "call-1",
                                }
                            ],
                        )
                    ]
                },
                "tools": {
                    "messages": [
                        SimpleNamespace(
                            type="tool",
                            content="found:A123",
                            tool_call_id="call-1",
                            name="lookup_order",
                        )
                    ]
                },
            },
        }
        yield {
            "type": "messages",
            "data": (SimpleNamespace(text="refund approved", content="refund approved"), {}),
        }
        yield {"type": "custom", "data": {"guardrail": "passed"}}


class FakeAsyncEventStreamAgent(FakeNativeAgent):
    async def astream_events(self, payload, config=None, version=None):
        del payload, config
        assert version == "v3"
        yield {
            "type": "messages",
            "data": (SimpleNamespace(text="async refund update"), {}),
        }
        yield {"type": "custom", "data": {"source": "async"}}


class FakeContextAgent(FakeNativeAgent):
    def invoke(self, payload, config=None, *, context=None):
        self.last_config = config
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"tenant={context['tenant']}; payload_context={'context' in payload}",
                }
            ]
        }


def fake_create_agent(**kwargs):
    return FakeNativeAgent(**kwargs)


def test_adapter_compiles_and_runs_native_agent(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()

    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return order_id

    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="openai/gpt-5",
        tools=[],
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert isinstance(compiled, CompiledLangChainAgent)
    assert compiled.native_agent.kwargs["name"] == "support_agent"
    assert compiled.native_agent.kwargs["model"] == "openai:gpt-5"
    assert compiled.native_agent.kwargs["system_prompt"] == "Echo the user request."
    assert result.backend == "langchain"
    assert result.output == "native langchain: hello"
    assert [event.type for event in result.events] == ["message", "complete"]
    assert result.metadata["run_diagnostics"]["messages_count"] == 1
    assert result.metadata["run_diagnostics"]["event_counts"] == {"message": 1, "complete": 1}


def test_stream_normalizes_interrupt_envelope_as_workflow_event() -> None:
    events = list(
        _normalize_native_events(
            {"type": "updates", "__interrupt__": [{"action": "approve"}]},
            backend="langchain",
        )
    )

    assert events[0].type == "workflow"
    assert events[0].data == {
        "phase": "interrupted",
        "interrupts": [{"action": "approve"}],
    }


def test_adapter_preserves_runtime_config_metadata(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Echo the user request.",
            model="openai/gpt-5",
        ),
        callbacks=["langsmith"],
        metadata={"owner": "support"},
    )

    compiled = adapter.compile(agent)
    result = adapter.run(
        compiled,
        RunInput(
            input="hello",
            metadata={"request_id": "req-1"},
            session_id="thread-1",
        ),
    )

    assert compiled.native_agent.last_config == {
        "callbacks": ["langsmith"],
        "configurable": {"thread_id": "thread-1"},
        "metadata": {
            "owner": "support",
            "request_id": "req-1",
            "session_id": "thread-1",
        },
    }
    assert result.metadata["runtime_config"] == compiled.native_agent.last_config
    assert result.metadata["run_diagnostics"]["runtime_config"] == compiled.native_agent.last_config
    assert result.metadata["extension_config"]["callbacks"] == ["langsmith"]
    assert result.metadata["native_agent_type"] == "FakeNativeAgent"


def test_adapter_passes_runtime_context_through_native_context_channel(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=lambda **kwargs: FakeContextAgent(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="context_agent",
            instructions="Use tenant context.",
            model="openai/gpt-5",
        ),
        context_schema=object(),
    )

    result = Adapter().run(
        Adapter().compile(agent),
        RunInput(input="hello", context={"tenant": "support"}),
    )

    assert result.output == "tenant=support; payload_context=False"


def test_adapter_connects_langsmith_context_and_langfuse_callback(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    langfuse_handler = object()
    monkeypatch.setitem(
        sys.modules,
        "langfuse.langchain",
        SimpleNamespace(CallbackHandler=lambda: langfuse_handler),
    )
    tracing_calls = []

    @contextmanager
    def tracing_context(**kwargs):
        tracing_calls.append(kwargs)
        yield

    monkeypatch.setitem(
        sys.modules,
        "langsmith.run_helpers",
        SimpleNamespace(tracing_context=tracing_context),
    )
    adapter = Adapter()
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="observed_agent",
            instructions="Reply to the user.",
            model="agentbridge/offline",
        ),
        observability={
            "tags": ["test"],
            "run_name": "observed-run",
            "langsmith": {"enabled": True, "project_name": "agentbridge-tests"},
            "langfuse": {"enabled": True, "user_id": "user-1"},
        },
    )

    result = adapter.run(adapter.compile(agent), RunInput(input="hello", session_id="session-1"))

    runtime_config = result.metadata["runtime_config"]
    assert runtime_config["callbacks"] == ["object"]
    assert runtime_config["tags"] == ["test"]
    assert runtime_config["run_name"] == "observed-run"
    assert runtime_config["metadata"]["langfuse_user_id"] == "user-1"
    assert runtime_config["metadata"]["langfuse_session_id"] == "session-1"
    assert tracing_calls == [
        {
            "project_name": "agentbridge-tests",
            "tags": ["test"],
            "metadata": {},
            "enabled": True,
        }
    ]


def test_adapter_passes_native_langfuse_trace_options(monkeypatch) -> None:
    captured = {}

    def callback_handler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "langfuse.langchain",
        SimpleNamespace(CallbackHandler=callback_handler),
    )
    agent = LangChainExtension.with_config(
        AgentSpec(name="trace_agent", instructions="Reply.", model="agentbridge/offline"),
        observability={
            "langfuse": {
                "enabled": True,
                "release": "v1",
                "environment": "test",
                "session_id": "session-1",
                "trace_id": "trace-1",
                "user_id": "user-1",
            }
        },
    )
    from agentbridge.observability import callbacks_for_config

    callbacks_for_config(agent.backend_config["langchain"])
    assert captured == {
        "release": "v1",
        "environment": "test",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "user_id": "user-1",
    }
def test_adapter_forwards_langchain_extension_surface(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()
    checkpointer = object()
    store = object()
    cache = object()
    state_schema = object()
    context_schema = object()
    transformer = object()
    middleware = object()
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Echo the user request.",
            model="openai/gpt-5",
        ),
        middleware=[middleware],
        memory="conversation_buffer",
        retrievers=["policy_docs"],
        checkpointer=checkpointer,
        store=store,
        interrupt_before=["tools"],
        interrupt_after=["model"],
        cache=cache,
        state_schema=state_schema,
        context_schema=context_schema,
        transformers=[transformer],
        debug=True,
        native_options={"future_option": "enabled"},
    )

    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert compiled.native_agent.kwargs["middleware"] == [middleware]
    assert compiled.native_agent.kwargs["checkpointer"] is checkpointer
    assert compiled.native_agent.kwargs["store"] is store
    assert compiled.native_agent.kwargs["interrupt_before"] == ["tools"]
    assert compiled.native_agent.kwargs["interrupt_after"] == ["model"]
    assert compiled.native_agent.kwargs["cache"] is cache
    assert compiled.native_agent.kwargs["state_schema"] is state_schema
    assert compiled.native_agent.kwargs["context_schema"] is context_schema
    assert compiled.native_agent.kwargs["transformers"] == [transformer]
    assert compiled.native_agent.kwargs["debug"] is True
    assert compiled.native_agent.kwargs["future_option"] == "enabled"
    assert result.metadata["extension_summary"] == {
        "agent_type": None,
        "prompt_template": False,
        "middleware_count": 1,
        "callbacks_count": 0,
        "memory": {
            "requested": "conversation_buffer",
            "native_checkpointer": True,
        },
        "retrievers": {
            "requested": ["policy_docs"],
            "native_store": True,
        },
        "applied_native_options": [
            "checkpointer",
            "store",
            "interrupt_before",
            "interrupt_after",
            "cache",
            "state_schema",
            "context_schema",
            "transformers",
            "debug",
        ],
        "native_options_count": 1,
        "mcp_tools_count": 0,
    }


def test_native_options_cannot_override_agent_identity(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )

    agent = LangChainExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Echo the user request.",
            model="openai/gpt-5",
        ),
        native_options={"name": "unexpected"},
    )

    try:
        Adapter().compile(agent)
    except ValueError as exc:
        assert "name" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("reserved native option was accepted")


def test_adapter_passes_native_mcp_tools_without_hard_dependency(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    native_mcp_tool = SimpleNamespace(name="mcp_lookup", description="MCP lookup")
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="mcp_agent",
            instructions="Use the MCP tool.",
            model="openai/gpt-5",
        ),
        mcp_tools=[native_mcp_tool],
    )

    compiled = Adapter().compile(agent)

    assert compiled.native_agent.kwargs["tools"] == [native_mcp_tool]
    assert compiled.config["mcp_tools"] == [native_mcp_tool]


def test_adapter_builds_openai_compatible_native_model_from_options(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="nim_agent",
            instructions="Reply to the user.",
            model="openai/deepseek-ai/deepseek-v4-flash-0731",
        ),
        model_provider="openai",
        model_options={
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "secret-key",
        },
    )

    adapter = Adapter()
    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="hello"))

    native_model = compiled.native_agent.kwargs["model"]
    assert native_model.kwargs == {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "secret-key",
        "model": "deepseek-ai/deepseek-v4-flash-0731",
    }
    assert result.metadata["extension_config"]["model_options"]["api_key"] == "[redacted]"


def test_adapter_builds_local_ollama_model_from_options(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )

    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        SimpleNamespace(ChatOllama=FakeChatOllama),
    )
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="local_agent",
            instructions="Reply to the user.",
            model="ollama/llama3.2",
        ),
        model_provider="ollama",
        model_options={"base_url": "http://localhost:11434", "temperature": 0},
    )

    compiled = Adapter().compile(agent)
    native_model = compiled.native_agent.kwargs["model"]
    assert native_model.kwargs == {
        "base_url": "http://localhost:11434",
        "temperature": 0,
        "model": "llama3.2",
    }


def test_adapter_lazily_builds_provider_specific_model_from_options(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        SimpleNamespace(ChatAnthropic=FakeChatAnthropic),
    )
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="anthropic_agent",
            instructions="Reply to the user.",
            model="anthropic/claude-sonnet-4-5",
        ),
        model_provider="anthropic",
        model_options={"temperature": 0.2, "api_key": "secret-key"},
    )

    compiled = Adapter().compile(agent)
    native_model = compiled.native_agent.kwargs["model"]

    assert native_model.kwargs == {
        "temperature": 0.2,
        "api_key": "secret-key",
        "model": "claude-sonnet-4-5",
    }


def test_adapter_accepts_any_native_langchain_model_object(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    native_model = object()
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="provider_neutral_agent",
            instructions="Reply to the user.",
            model="vendor/custom-model",
        ),
        model=native_model,
    )

    compiled = Adapter().compile(agent)

    assert compiled.native_agent.kwargs["model"] is native_model


def test_adapter_summarizes_native_langchain_retrievers(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()
    retriever = SimpleNamespace(name="refund_policy_retriever")
    checkpointer = object()
    store = object()
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Use policy docs.",
            model="openai/gpt-5",
        ),
        memory="langgraph_in_memory_checkpointer",
        retrievers=[retriever],
        checkpointer=checkpointer,
        store=store,
    )

    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert compiled.native_agent.kwargs["checkpointer"] is checkpointer
    assert compiled.native_agent.kwargs["store"] is store
    assert result.metadata["extension_summary"]["memory"] == {
        "requested": "langgraph_in_memory_checkpointer",
        "native_checkpointer": True,
    }
    assert result.metadata["extension_summary"]["retrievers"] == {
        "requested": [{"name": "refund_policy_retriever"}],
        "native_store": True,
    }


def test_adapter_runs_offline_model_through_create_agent() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="agentbridge/offline",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "langchain"
    assert result.output == "offline response: hello"
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_runs_offline_tool_loop_through_create_agent() -> None:
    adapter = Adapter()

    def lookup_order(order_id: str) -> str:
        """Look up an order."""

        return f"found:{order_id}"

    spec = AgentSpec(
        name="support_agent",
        instructions="Use the lookup tool.",
        model="agentbridge/offline",
        tools=[ToolSpec.from_function(lookup_order)],
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="A123"))

    event_types = [event.type for event in result.events]
    assert result.output == "offline tool result: found:A123"
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert event_types[-1] == "complete"


def test_adapter_streams_langchain_event_stream_shapes(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=lambda **kwargs: FakeEventStreamAgent(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Use tools.",
        model="openai/gpt-5",
    )

    compiled = adapter.compile(spec)
    events = list(adapter.stream(compiled, RunInput(input="check A123")))

    assert [event.type for event in events] == [
        "tool_call",
        "tool_call",
        "tool_result",
        "message",
        "workflow",
    ]
    assert events[0].data["delta"] is True
    assert events[0].data["metadata"] == {"lc_agent_name": "support_agent"}
    assert events[1].data["args"] == {"order_id": "A123"}
    assert events[1].metadata["source"] == "model"
    assert events[2].data["content"] == "found:A123"
    assert events[2].metadata["source"] == "tools"
    assert events[3].data["content"] == "refund approved"
    assert events[4].data["data"] == {"guardrail": "passed"}


def test_adapter_streams_native_async_langchain_events(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=lambda **kwargs: FakeAsyncEventStreamAgent(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()
    spec = AgentSpec(
        name="async_support_agent",
        instructions="Use async events.",
        model="openai/gpt-5",
    )

    async def collect():
        compiled = adapter.compile(spec)
        events = []
        async for event in adapter.astream(compiled, RunInput(input="check A123")):
            events.append(event)
        return events

    import asyncio

    events = asyncio.run(collect())
    assert [event.type for event in events] == ["message", "workflow"]
    assert events[0].data["content"] == "async refund update"


def test_adapter_streams_classic_langchain_stream_updates(monkeypatch) -> None:
    class ClassicStreamAgent(FakeNativeAgent):
        def stream(self, payload, config=None, stream_mode=None, version=None):
            del payload, config
            assert stream_mode == ["messages", "updates", "custom"]
            assert version == "v2"
            yield {
                "model": {
                    "messages": [
                        {
                            "type": "ai",
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "lookup_order",
                                    "args": {"order_id": "A123"},
                                }
                            ],
                        }
                    ]
                }
            }
            yield {"messages": [{"content": "stream chunk"}]}

    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=lambda **kwargs: ClassicStreamAgent(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(StructuredTool=FakeStructuredTool),
    )
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Use tools.",
        model="openai/gpt-5",
    )

    compiled = adapter.compile(spec)
    events = list(adapter.stream(compiled, RunInput(input="check A123")))

    assert [event.type for event in events] == ["tool_call", "message"]
    assert events[0].data["name"] == "lookup_order"
    assert events[0].metadata["source"] == "model"
    assert events[1].data["content"] == "stream chunk"


def test_adapter_runs_offline_structured_output_through_create_agent() -> None:
    class Decision(BaseModel):
        eligible: bool = True
        reason: str = "ok"

    adapter = Adapter()
    spec = AgentSpec(
        name="decision_agent",
        instructions="Return a decision.",
        model="agentbridge/offline",
        output_type=Decision,
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="decide"))

    event_types = [event.type for event in result.events]
    assert result.backend == "langchain"
    assert isinstance(result.output, Decision)
    assert result.output.eligible is True
    assert result.output.reason == "ok"
    assert "tool_call" in event_types
    assert event_types[-1] == "complete"


def test_adapter_capabilities_mark_structured_output_full() -> None:
    capabilities = Adapter().capabilities()

    assert capabilities.status("structured_output") == "full"
    assert capabilities.status("observability.diagnostics") == "full"


def test_adapter_summarizes_agentcore_bindings():
    agent = LangChainExtension.with_config(
        AgentSpec(name="support", instructions="Help.", model="agentbridge/offline"),
        agentcore={"memory_id": "memory-1", "gateway_url": "https://gateway.example"},
    )
    result = Adapter().run(Adapter().compile(agent), RunInput(input="hello"))
    assert result.metadata["extension_summary"]["agentcore"]["memory_id"] == "memory-1"
