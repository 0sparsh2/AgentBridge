"""LangChain-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class LangChainConfig(BaseModel):
    """AgentBridge config for LangChain-native behavior."""

    agent_type: str | None = None
    prompt_template: str | None = None
    middleware: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    memory: str | None = None
    retrievers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LangChainExtension(FrameworkExtension):
    """Extension namespace for LangChain behavior outside LangGraph."""

    framework = "langchain"

    @staticmethod
    def config(
        *,
        agent_type: str | None = None,
        prompt_template: str | None = None,
        middleware: list[str] | None = None,
        callbacks: list[str] | None = None,
        memory: str | None = None,
        retrievers: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable LangChain adapter configuration."""

        return LangChainConfig(
            agent_type=agent_type,
            prompt_template=prompt_template,
            middleware=middleware or [],
            callbacks=callbacks or [],
            memory=memory,
            retrievers=retrievers or [],
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with LangChain config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["langchain"] = LangChainExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
