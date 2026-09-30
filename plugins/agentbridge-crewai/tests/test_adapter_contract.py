from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput
from agentbridge_crewai.adapter import CrewAIAdapter


class _Native:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Crew(_Native):
    def kickoff(self, *, inputs):
        return f"{self.kwargs['process']}:{inputs['input']}"


def test_crewai_adapter_executes_native_role_task_crew_contract(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "crewai",
        SimpleNamespace(
            Agent=lambda **kwargs: _Native(**kwargs),
            Task=lambda **kwargs: _Native(**kwargs),
            Crew=lambda **kwargs: _Crew(**kwargs),
            Process=SimpleNamespace(sequential="native-sequential"),
        ),
    )
    spec = AgentSpec(
        name="refund_agent",
        instructions="Decide refunds.",
        model="agentbridge/offline",
        backend_config={
            "crewai": {
                "task_description": "Review {input}",
                "expected_output": "Decision",
            }
        },
    )
    result = CrewAIAdapter().run(CrewAIAdapter().compile(spec), RunInput(input="A123"))
    assert result.output == "native-sequential:A123"
    assert result.metadata["role"] == "refund_agent"
    assert result.events[-1].type == "complete"
