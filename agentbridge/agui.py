"""Helpers for converting AgentBridge events into AG-UI-shaped events."""

from __future__ import annotations

from typing import Any

from agentbridge.types import AgentEvent


def event_to_agui(event: AgentEvent) -> dict[str, Any]:
    """Convert an AgentBridge event into an AG-UI-shaped dictionary.

    This is intentionally lightweight. It preserves enough shape for examples
    without claiming to implement the full AG-UI protocol server in v0.
    """

    event_type_map = {
        "message": "TEXT_MESSAGE_CONTENT",
        "tool_call": "TOOL_CALL_START",
        "tool_result": "TOOL_CALL_RESULT",
        "error": "RUN_ERROR",
        "complete": "RUN_FINISHED",
    }
    return {
        "type": event_type_map[event.type],
        "source": "agentbridge",
        "backend": event.backend,
        "data": event.data,
        "metadata": event.metadata,
    }
