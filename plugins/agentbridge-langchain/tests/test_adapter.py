from __future__ import annotations

import sys
from types import SimpleNamespace

from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge_langchain.adapter import Adapter, CompiledLangChainAgent


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
                {"role": "assistant", "content": f"native langchain: {payload['messages'][0]['content']}"}
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
