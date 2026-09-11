"""Pydantic AI-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension
from agentbridge.types import AgentSpec


class PydanticAIConfig(BaseModel):
    """AgentBridge config for Pydantic AI-native behavior."""

    retries: int | None = Field(default=None, ge=0)
    tool_timeout: float | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    custom_output_text: str | None = None
    custom_output_args: Any | None = None


class PydanticAIExtension(FrameworkExtension):
    """Extension namespace for Pydantic AI-native behavior."""

    framework = "pydantic_ai"

    @staticmethod
    def config(
        *,
        retries: int | None = None,
        tool_timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
        custom_output_text: str | None = None,
        custom_output_args: Any | None = None,
    ) -> dict[str, Any]:
        """Build serializable Pydantic AI adapter configuration."""

        return PydanticAIConfig(
            retries=retries,
            tool_timeout=tool_timeout,
            metadata=metadata or {},
            custom_output_text=custom_output_text,
            custom_output_args=custom_output_args,
        ).model_dump(exclude_none=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with Pydantic AI extension config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["pydantic_ai"] = PydanticAIExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})

    def validation_retries(self) -> dict[str, Any]:
        """Return metadata for validation retry configuration."""

        raw = self.require_raw()
        return {
            "framework": self.framework,
            "raw_type": type(raw).__name__,
            "message": "Use retries=... in PydanticAIExtension.config().",
        }

    def dependency_injection(self) -> None:
        """Placeholder for future dependency/deps helpers."""

        raise UnsupportedExtension("Pydantic AI dependency injection extension is not implemented yet.")
