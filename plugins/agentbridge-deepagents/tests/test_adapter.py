from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge_deepagents.adapter import Adapter


class FakeStructuredTool:
    @staticmethod
    def from_function(**kwargs):
        return SimpleNamespace(**kwargs)


class FakeDeepAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.context = None

    def invoke(self, payload, config=None, context=None):
        self.context = context
        return {"messages": [{"content": f"tenant={context['tenant']}; input={payload['messages'][0]['content']}"}]}

    def stream(self, payload, config=None, context=None):
        del payload, config, context
        yield {"output": "streamed"}


def test_deepagents_adapter_maps_native_harness_options(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "deepagents",
        SimpleNamespace(create_deep_agent=lambda **kwargs: FakeDeepAgent(**kwargs)),
    )
    monkeypatch.setitem(sys.modules, "langchain_core.tools", SimpleNamespace(StructuredTool=FakeStructuredTool))

    def lookup(order_id: str) -> str:
        """Look up an order."""
        return order_id

    agent = AgentSpec(
        name="research_agent",
        instructions="Research carefully.",
        model="openai/gpt-5",
        tools=[ToolSpec.from_function(lookup)],
        backend_config={
            "deepagents": {
                "skills": ["/skills/project/"],
                "memory": ["/memory/AGENTS.md"],
                "permissions": [{"mode": "deny", "path": "/secrets/**"}],
                "native_options": {"custom_profile": "research"},
            }
        },
    )
    adapter = Adapter()
    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="A123", context={"tenant": "support"}))

    assert compiled.native_agent.kwargs["model"] == "openai:gpt-5"
    assert compiled.native_agent.kwargs["skills"] == ["/skills/project/"]
    assert compiled.native_agent.kwargs["custom_profile"] == "research"
    assert result.output == "tenant=support; input=A123"


def test_deepagents_adapter_normalizes_streaming(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepagents", SimpleNamespace(create_deep_agent=lambda **kwargs: FakeDeepAgent()))
    monkeypatch.setitem(sys.modules, "langchain_core.tools", SimpleNamespace(StructuredTool=FakeStructuredTool))
    agent = AgentSpec(name="stream_agent", instructions="Stream.", model="openai/gpt-5")
    events = list(Adapter().stream(Adapter().compile(agent), RunInput(input="hello")))

    assert events[0].type == "workflow"
    assert events[-1].type == "complete"
    assert events[-1].data["output"] == "streamed"


def test_deepagents_adapter_rejects_reserved_native_options(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepagents", SimpleNamespace(create_deep_agent=lambda **kwargs: FakeDeepAgent(**kwargs)))
    monkeypatch.setitem(sys.modules, "langchain_core.tools", SimpleNamespace(StructuredTool=FakeStructuredTool))
    agent = AgentSpec(
        name="reserved_agent",
        instructions="Do not override identity.",
        model="openai/gpt-5",
        backend_config={"deepagents": {"native_options": {"name": "wrong"}}},
    )

    try:
        Adapter().compile(agent)
    except ValueError as exc:
        assert "name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("reserved native option was accepted")
