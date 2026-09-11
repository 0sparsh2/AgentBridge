"""Pydantic AI-specific extension helpers."""

from __future__ import annotations

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension


class PydanticAIExtension(FrameworkExtension):
    """Extension namespace for Pydantic AI-native behavior."""

    framework = "pydantic_ai"

    def validation_retries(self) -> None:
        """Placeholder for future structured-output validation retry helpers."""

        raise UnsupportedExtension("Pydantic AI validation retry extension is not implemented yet.")

    def dependency_injection(self) -> None:
        """Placeholder for future dependency/deps helpers."""

        raise UnsupportedExtension("Pydantic AI dependency injection extension is not implemented yet.")
