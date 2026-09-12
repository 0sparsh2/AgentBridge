"""Strands Agents-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class StrandsConfig(BaseModel):
    """AgentBridge config for Strands-native behavior."""

    conversation_manager: str | None = None
    hooks: list[str] = Field(default_factory=list)
    mcp_clients: list[str] = Field(default_factory=list)
    trace_attributes: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[str] = Field(default_factory=list)
    deployment_target: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrandsExtension(FrameworkExtension):
    """Extension namespace for Strands Agents behavior."""

    framework = "strands"

    @staticmethod
    def config(
        *,
        conversation_manager: str | None = None,
        hooks: list[str] | None = None,
        mcp_clients: list[str] | None = None,
        trace_attributes: dict[str, Any] | None = None,
        guardrails: list[str] | None = None,
        deployment_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable Strands adapter configuration."""

        return StrandsConfig(
            conversation_manager=conversation_manager,
            hooks=hooks or [],
            mcp_clients=mcp_clients or [],
            trace_attributes=trace_attributes or {},
            guardrails=guardrails or [],
            deployment_target=deployment_target,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with Strands config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["strands"] = StrandsExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
