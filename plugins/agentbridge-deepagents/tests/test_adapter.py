from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge_deepagents.adapter import Adapter


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_config = None

    def invoke(self, payload, config=None):
        self.last_config = config
        return {"messages": [{"content": payload["messages"][0]["content"]}]}

    def stream(self, payload, config=None, **kwargs):
        del payload, config, kwargs
        yield {"content": "chunk"}


def test_deep_agents_adapter_maps_core_and_native_options(monkeypatch):
    fake_module = SimpleNamespace(create_deep_agent=lambda **kwargs: FakeAgent(**kwargs))
    monkeypatch.setitem(sys.modules, "deepagents", fake_module)
    adapter = Adapter()

    def lookup(order_id: str) -> str:
        return order_id

    agent = AgentSpec(
        name="refunds",
        instructions="Decide eligibility.",
        model="openai/gpt-5",
        tools=[ToolSpec.from_function(lookup)],
        backend_config={
            "deepagents": {
                "memory": ["AGENTS.md"],
                "skills": ["skills/"],
                "native_options": {"future_option": True},
                "protocols": ["ag_ui", "a2a"],
                "sandbox": {"provider": "modal", "image": "python:3.12"},
            }
        },
    )

    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="A123", session_id="s-1"))

    assert compiled.native_agent.kwargs["model"] == "openai:gpt-5"
    assert compiled.native_agent.kwargs["tools"][0] is lookup
    assert compiled.native_agent.kwargs["memory"] == ["AGENTS.md"]
    assert compiled.native_agent.kwargs["future_option"] is True
    assert result.output == "A123"
    assert result.metadata["native_options_count"] == 1
    assert result.metadata["protocols"] == ["ag_ui", "a2a"]
    assert result.metadata["sandbox"]["provider"] == "modal"


def test_deep_agents_adapter_rejects_identity_override(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "deepagents",
        SimpleNamespace(create_deep_agent=lambda **kwargs: FakeAgent(**kwargs)),
    )
    agent = AgentSpec(
        name="refunds",
        instructions="Decide eligibility.",
        model="openai/gpt-5",
        backend_config={"deepagents": {"native_options": {"name": "wrong"}}},
    )

    try:
        Adapter().compile(agent)
    except ValueError as exc:
        assert "name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("reserved native option was accepted")
