from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentbridge import AgentEvent, AgentSpec, RunResult, ToolSpec


def sample_tool(order_id: str, dry_run: bool = True) -> str:
    """Check an order."""

    return f"{order_id}:{dry_run}"


def test_agent_spec_rejects_blank_values() -> None:
    with pytest.raises(ValidationError):
        AgentSpec(name="", instructions="Do work", model="openai/gpt-5")

    with pytest.raises(ValidationError):
        AgentSpec(name="agent", instructions=" ", model="openai/gpt-5")


def test_tool_spec_from_function_extracts_schema() -> None:
    tool = ToolSpec.from_function(sample_tool)

    assert tool.name == "sample_tool"
    assert tool.description == "Check an order."
    assert tool.input_schema["properties"]["order_id"]["type"] == "string"
    assert tool.input_schema["properties"]["dry_run"]["type"] == "boolean"
    assert tool.input_schema["required"] == ["order_id"]
    assert tool.call({"order_id": "A123"}) == "A123:True"


def test_result_and_event_serialize_to_json() -> None:
    event = AgentEvent(type="complete", backend="mock", data={"output": "ok"})
    result = RunResult(output="ok", backend="mock", events=[event], raw=object())

    dumped = result.model_dump()
    assert dumped["output"] == "ok"
    assert dumped["events"][0]["type"] == "complete"
    assert "raw" not in dumped
