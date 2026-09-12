from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput
from agentbridge_google_adk.adapter import Adapter, CompiledGoogleADKAgent


class FakeADKAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFunctionTool:
    def __init__(self, func):
        self.func = func


class FakeSessionService:
    pass


class FakePart:
    def __init__(self, text=None):
        self.text = text

    @classmethod
    def from_text(cls, *, text):
        return cls(text=text)


class FakeContent:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []


class FakeRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, *, user_id, session_id, new_message, state_delta=None, run_config=None):
        yield SimpleNamespace(
            content=FakeContent(
                role="model",
                parts=[FakePart(text=f"native google adk: {new_message.parts[0].text}")],
            )
        )


def install_fake_google_adk(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.adk.agents", SimpleNamespace(Agent=FakeADKAgent))
    monkeypatch.setitem(sys.modules, "google.adk.runners", SimpleNamespace(Runner=FakeRunner))
    monkeypatch.setitem(
        sys.modules,
        "google.adk.sessions",
        SimpleNamespace(InMemorySessionService=FakeSessionService),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.adk.tools.function_tool",
        SimpleNamespace(FunctionTool=FakeFunctionTool),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.genai.types",
        SimpleNamespace(Content=FakeContent, Part=FakePart),
    )


def test_adapter_compiles_and_runs_native_agent(monkeypatch) -> None:
    install_fake_google_adk(monkeypatch)
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="mock/model",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert isinstance(compiled, CompiledGoogleADKAgent)
    assert compiled.native_agent.kwargs["name"] == "support_agent"
    assert compiled.native_agent.kwargs["instruction"] == "Echo the user request."
    assert compiled.runner.kwargs["app_name"] == "support_agent"
    assert result.backend == "google_adk"
    assert result.output == "native google adk: hello"
    assert [event.type for event in result.events] == ["message", "complete"]
