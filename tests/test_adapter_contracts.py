from __future__ import annotations

import pytest

from agentbridge import AgentSpec, RunInput, ToolSpec, get_adapter
from agentbridge.errors import MissingDependencyError


def lookup_order(order_id: str) -> str:
    """Look up an order."""

    return f"found:{order_id}"


@pytest.mark.parametrize("backend", ["mock", "langgraph", "pydantic_ai"])
def test_verified_adapters_execute_sync_tools_when_marked_full(backend: str) -> None:
    adapter = get_adapter(backend)
    capabilities = adapter.capabilities()
    if capabilities.status("tools.sync") != "full":
        pytest.skip(f"{backend} does not advertise full sync tool support")

    model = "test" if backend == "pydantic_ai" else "openai/gpt-5"
    agent = AgentSpec(
        name=f"{backend}_agent",
        instructions="Use the lookup tool.",
        model=model,
        tools=[ToolSpec.from_function(lookup_order)],
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip(f"{backend} optional dependency is not installed")

    result = adapter.run(compiled, RunInput(input="A123"))
    event_types = [event.type for event in result.events]

    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert result.output


def test_crewai_adapter_is_not_builtin() -> None:
    with pytest.raises(Exception, match="Unknown backend"):
        get_adapter("crewai")
