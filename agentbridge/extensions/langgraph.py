"""LangGraph-specific extension helpers."""

from __future__ import annotations

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension


class LangGraphExtension(FrameworkExtension):
    """Extension namespace for LangGraph-native behavior."""

    framework = "langgraph"

    def checkpointing(self) -> None:
        """Placeholder for future checkpoint/resume helpers."""

        raise UnsupportedExtension("LangGraph checkpointing extension is not implemented yet.")

    def conditional_routing(self) -> None:
        """Placeholder for future graph routing helpers."""

        raise UnsupportedExtension("LangGraph conditional routing extension is not implemented yet.")
