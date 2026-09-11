from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from agentbridge.adapters import BackendAdapter
from agentbridge import AgentSpec, RunInput
from agentbridge.extensions.crewai import CrewAIExtension


def test_crewai_plugin_scaffold_exposes_adapter_without_crewai_installed() -> None:
    module = _load_crewai_adapter_module("agentbridge_crewai_test_adapter")

    adapter_type = module.CrewAIAdapter

    assert issubclass(adapter_type, BackendAdapter)
    assert adapter_type.backend_name == "crewai"
    assert adapter_type().capabilities().status("workflow.roles_tasks") == "full"


def test_crewai_plugin_maps_extension_config_to_native_shapes(monkeypatch) -> None:
    calls = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            calls["agent"] = kwargs

    class FakeTask:
        def __init__(self, **kwargs):
            calls["task"] = kwargs

    class FakeCrew:
        def __init__(self, **kwargs):
            calls["crew"] = kwargs

        def kickoff(self, inputs):
            calls["kickoff"] = inputs
            return "crew result"

    fake_crewai = types.ModuleType("crewai")
    fake_crewai.Agent = FakeAgent
    fake_crewai.Task = FakeTask
    fake_crewai.Crew = FakeCrew
    fake_crewai.Process = types.SimpleNamespace(sequential="sequential", hierarchical="hierarchical")
    monkeypatch.setitem(sys.modules, "crewai", fake_crewai)

    module = _load_crewai_adapter_module("agentbridge_crewai_fake_runtime_adapter")
    adapter = module.CrewAIAdapter()
    agent = CrewAIExtension.with_config(
        AgentSpec(
            name="refund_agent",
            instructions="Resolve refunds.",
            model="openai/gpt-5",
        ),
        role="Refund specialist",
        goal="Resolve refund requests",
        backstory="Expert in policy.",
        task_description="Review {input}",
        expected_output="Refund decision",
        process="hierarchical",
        verbose=True,
        allow_delegation=True,
        memory=True,
        human_input=True,
        metadata={"owner": "support"},
    )

    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="Order A123", context={"tenant": "support"}))

    assert calls["agent"]["role"] == "Refund specialist"
    assert calls["agent"]["goal"] == "Resolve refund requests"
    assert calls["agent"]["allow_delegation"] is True
    assert calls["task"]["description"] == "Review {input}"
    assert calls["task"]["expected_output"] == "Refund decision"
    assert calls["task"]["human_input"] is True
    assert calls["crew"]["process"] == "hierarchical"
    assert calls["crew"]["memory"] is True
    assert calls["kickoff"] == {"input": "Order A123", "tenant": "support"}
    assert result.output == "crew result"
    assert result.metadata["role"] == "Refund specialist"
    assert result.metadata["process"] == "hierarchical"
    assert result.metadata["memory"] is True
    assert result.metadata["human_input"] is True
    assert result.metadata["extension_metadata"] == {"owner": "support"}


def _load_crewai_adapter_module(name: str):
    adapter_path = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "agentbridge-crewai"
        / "agentbridge_crewai"
        / "adapter.py"
    )
    spec = importlib.util.spec_from_file_location(name, adapter_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
