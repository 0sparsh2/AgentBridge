from __future__ import annotations

from agentbridge_google_adk.adapter import Adapter
from agentbridge import AgentSpec, RunInput


def test_adapter_runs_template_agent() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="template_agent",
        instructions="Echo the user request.",
        model="mock/model",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "google_adk"
    assert "hello" in result.output
    assert [event.type for event in result.events] == ["message", "complete"]
