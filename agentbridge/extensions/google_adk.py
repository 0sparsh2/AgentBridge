"""Google ADK-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class GoogleADKConfig(BaseModel):
    """AgentBridge config for Google ADK-native behavior."""

    app_name: str | None = None
    session_service: str | None = None
    memory_service: str | None = None
    artifact_service: str | None = None
    sub_agents: list[str] = Field(default_factory=list)
    deployment_target: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoogleADKExtension(FrameworkExtension):
    """Extension namespace for Google ADK behavior."""

    framework = "google_adk"

    @staticmethod
    def config(
        *,
        app_name: str | None = None,
        session_service: str | None = None,
        memory_service: str | None = None,
        artifact_service: str | None = None,
        sub_agents: list[str] | None = None,
        deployment_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable Google ADK adapter configuration."""

        return GoogleADKConfig(
            app_name=app_name,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            sub_agents=sub_agents or [],
            deployment_target=deployment_target,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with Google ADK config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["google_adk"] = GoogleADKExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
