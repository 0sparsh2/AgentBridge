from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge_strands.adapter import Adapter, CompiledStrandsAgent


class FakeStrandsAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(
            message={"content": [{"text": f"native strands: {prompt}"}]},
            metrics={"requests": 1},
            stop_reason="end_turn",
            structured_output=None,
            interrupts=None,
            checkpoint=None,
        )


def fake_tool(func, **kwargs):
    return SimpleNamespace(func=func, kwargs=kwargs)


def test_adapter_compiles_and_runs_native_agent(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "strands",
        SimpleNamespace(Agent=FakeStrandsAgent, tool=fake_tool),
    )
    adapter = Adapter()

    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return order_id

    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="mock/model",
        tools=[],
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert isinstance(compiled, CompiledStrandsAgent)
    assert compiled.native_agent.kwargs["name"] == "support_agent"
    assert compiled.native_agent.kwargs["system_prompt"] == "Echo the user request."
    assert result.backend == "strands"
    assert result.output == "native strands: hello"
    assert result.usage == {"requests": 1}
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_runs_offline_model_through_native_agent() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="agentbridge/offline",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "strands"
    assert result.output == "offline response: hello"
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_runs_offline_tool_loop_through_native_agent() -> None:
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
