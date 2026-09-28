"""Deep Agents-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class DeepAgentsConfig(BaseModel):
    """Configuration forwarded to LangChain Deep Agents."""

    model_config = {"arbitrary_types_allowed": True}

    system_prompt: Any | None = None
    middleware: list[Any] = Field(default_factory=list)
    subagents: list[Any] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)
    permissions: list[Any] = Field(default_factory=list)
    backend: Any | None = None
    interrupt_on: dict[str, Any] = Field(default_factory=dict)
    response_format: Any | None = None
    state_schema: Any | None = None
    context_schema: Any | None = None
    checkpointer: Any | None = None
    store: Any | None = None
    debug: bool = False
    name: str | None = None
    cache: Any | None = None
    native_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)


class DeepAgentsExtension(FrameworkExtension):
    """Extension namespace for Deep Agents-native features."""

    framework = "deepagents"

    @staticmethod
    def config(**kwargs: Any) -> dict[str, Any]:
        return DeepAgentsConfig(**kwargs).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        backend_config = dict(spec.backend_config)
        backend_config["deepagents"] = DeepAgentsExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
