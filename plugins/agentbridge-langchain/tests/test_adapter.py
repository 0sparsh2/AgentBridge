from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge_langchain.adapter import Adapter, CompiledLangChainAgent


class FakeStructuredTool:
    @staticmethod
    def from_function(**kwargs):
        return SimpleNamespace(**kwargs)


class FakeNativeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, payload, config=None):
        return {
            "messages": [
                {"role": "assistant", "content": f"native langchain: {payload['messages'][0]['content']}"}
            ]
        }

    def stream(self, payload, config=None):
        yield {"messages": [{"content": "stream chunk"}]}


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
