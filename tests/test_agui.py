from __future__ import annotations

from agentbridge import AgentEvent, event_to_agui


def test_event_to_agui_shape() -> None:
    event = AgentEvent(type="tool_call", backend="mock", data={"name": "lookup"})

    agui_event = event_to_agui(event)

    assert agui_event["type"] == "TOOL_CALL_START"
    assert agui_event["source"] == "agentbridge"
    assert agui_event["backend"] == "mock"
    assert agui_event["data"]["name"] == "lookup"
