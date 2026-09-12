"""LangChain-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class LangChainConfig(BaseModel):
    """AgentBridge config for LangChain-native behavior."""

    model_config = {"arbitrary_types_allowed": True}

    agent_type: str | None = None
    prompt_template: str | None = None
    middleware: list[Any] = Field(default_factory=list)
    callbacks: list[Any] = Field(default_factory=list)
    memory: str | None = None
    retrievers: list[Any] = Field(default_factory=list)
    checkpointer: Any | None = None
    store: Any | None = None
    interrupt_before: list[str] | None = None
    interrupt_after: list[str] | None = None
    cache: Any | None = None
    state_schema: Any | None = None
    context_schema: Any | None = None
    transformers: list[Any] = Field(default_factory=list)
    debug: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LangChainExtension(FrameworkExtension):
    """Extension namespace for LangChain behavior outside LangGraph."""

    framework = "langchain"

    @staticmethod
    def config(
        *,
        agent_type: str | None = None,
        prompt_template: str | None = None,
        middleware: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        memory: str | None = None,
        retrievers: list[Any] | None = None,
        checkpointer: Any | None = None,
        store: Any | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        cache: Any | None = None,
        state_schema: Any | None = None,
        context_schema: Any | None = None,
        transformers: list[Any] | None = None,
        debug: bool | None = None,
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
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            cache=cache,
            state_schema=state_schema,
            context_schema=context_schema,
            transformers=transformers or [],
            debug=debug,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with LangChain config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["langchain"] = LangChainExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
