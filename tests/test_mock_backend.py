from __future__ import annotations

import pytest

from agentbridge import AgentSpec, ToolSpec, run_agent, stream_agent


def lookup(query: str) -> str:
    """Look up a query."""

    return f"found:{query}"


def test_mock_backend_runs_agent_with_tool() -> None:
    agent = AgentSpec(
        name="research_agent",
        instructions="Research things.",
        model="openai/gpt-5",
        tools=[ToolSpec.from_function(lookup)],
    )

    result = run_agent(agent, backend="mock", input="frameworks")

    assert result.backend == "mock"
    assert result.output["agent"] == "research_agent"
    assert result.output["tools"][0]["name"] == "lookup"
    assert result.events[-1].type == "complete"


def test_mock_backend_runs_agent_with_framework_alias() -> None:
    agent = AgentSpec(name="plain_agent", instructions="Reply.", model="openai/gpt-5")

    result = run_agent(agent, framework="mock", input="hello")

    assert result.backend == "mock"
    assert result.output["agent"] == "plain_agent"


def test_run_agent_rejects_conflicting_backend_and_framework() -> None:
    agent = AgentSpec(name="plain_agent", instructions="Reply.", model="openai/gpt-5")

    with pytest.raises(ValueError, match="backend or framework"):
        run_agent(agent, backend="mock", framework="langgraph", input="hello")


def test_mock_backend_streams_normalized_events() -> None:
    agent = AgentSpec(name="plain_agent", instructions="Reply.", model="openai/gpt-5")

    events = list(stream_agent(agent, framework="mock", input="hello"))

    assert [event.type for event in events] == ["message", "complete"]
