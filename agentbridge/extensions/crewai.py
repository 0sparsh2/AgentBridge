"""CrewAI-specific extension helpers."""

from __future__ import annotations

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension


class CrewAIExtension(FrameworkExtension):
    """Extension namespace for CrewAI-native behavior."""

    framework = "crewai"

    def crews(self) -> None:
        """Placeholder for future crew/team helpers."""

        raise UnsupportedExtension("CrewAI crew extension is not implemented yet.")

    def tasks(self) -> None:
        """Placeholder for future role/task helpers."""

        raise UnsupportedExtension("CrewAI task extension is not implemented yet.")
