"""OpenAI Agents SDK-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class OpenAIAgentsConfig(BaseModel):
    """AgentBridge config for OpenAI Agents-native behavior."""

    handoffs: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    tracing: bool = False
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAIAgentsExtension(FrameworkExtension):
    """Extension namespace for OpenAI Agents SDK behavior."""

    framework = "openai_agents"

    @staticmethod
    def config(
        *,
        handoffs: list[str] | None = None,
        guardrails: list[str] | None = None,
        tracing: bool = False,
        approval_policy: dict[str, Any] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable OpenAI Agents adapter configuration."""

        return OpenAIAgentsConfig(
            handoffs=handoffs or [],
            guardrails=guardrails or [],
            tracing=tracing,
            approval_policy=approval_policy or {},
            session_id=session_id,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with OpenAI Agents config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["openai_agents"] = OpenAIAgentsExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
