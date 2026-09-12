"""OpenAI Agents SDK-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class OpenAIAgentsConfig(BaseModel):
    """AgentBridge config for OpenAI Agents-native behavior."""

    model_config = {"arbitrary_types_allowed": True}

    handoff_description: str | None = None
    handoffs: list[Any] = Field(default_factory=list)
    mcp_servers: list[Any] = Field(default_factory=list)
    mcp_config: Any | None = None
    prompt: Any | None = None
    model_settings: Any | None = None
    input_guardrails: list[Any] = Field(default_factory=list)
    output_guardrails: list[Any] = Field(default_factory=list)
    guardrails: list[Any] = Field(default_factory=list)
    hooks: Any | None = None
    tool_use_behavior: Any | None = None
    reset_tool_choice: bool | None = None
    tracing: bool = False
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    context: Any | None = None
    max_turns: int | None = None
    run_hooks: Any | None = None
    run_config: Any | None = None
    error_handlers: Any | None = None
    previous_response_id: str | None = None
    auto_previous_response_id: bool | None = None
    conversation_id: str | None = None
    session: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAIAgentsExtension(FrameworkExtension):
    """Extension namespace for OpenAI Agents SDK behavior."""

    framework = "openai_agents"

    @staticmethod
    def config(
        *,
        handoff_description: str | None = None,
        handoffs: list[Any] | None = None,
        mcp_servers: list[Any] | None = None,
        mcp_config: Any | None = None,
        prompt: Any | None = None,
        model_settings: Any | None = None,
        input_guardrails: list[Any] | None = None,
        output_guardrails: list[Any] | None = None,
        guardrails: list[Any] | None = None,
        hooks: Any | None = None,
        tool_use_behavior: Any | None = None,
        reset_tool_choice: bool | None = None,
        tracing: bool = False,
        approval_policy: dict[str, Any] | None = None,
        session_id: str | None = None,
        context: Any | None = None,
        max_turns: int | None = None,
        run_hooks: Any | None = None,
        run_config: Any | None = None,
        error_handlers: Any | None = None,
        previous_response_id: str | None = None,
        auto_previous_response_id: bool | None = None,
        conversation_id: str | None = None,
        session: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable OpenAI Agents adapter configuration."""

        return OpenAIAgentsConfig(
            handoff_description=handoff_description,
            handoffs=handoffs or [],
            mcp_servers=mcp_servers or [],
            mcp_config=mcp_config,
            prompt=prompt,
            model_settings=model_settings,
            input_guardrails=input_guardrails or [],
            output_guardrails=output_guardrails or [],
            guardrails=guardrails or [],
            hooks=hooks,
            tool_use_behavior=tool_use_behavior,
            reset_tool_choice=reset_tool_choice,
            tracing=tracing,
            approval_policy=approval_policy or {},
            session_id=session_id,
            context=context,
            max_turns=max_turns,
            run_hooks=run_hooks,
            run_config=run_config,
            error_handlers=error_handlers,
            previous_response_id=previous_response_id,
            auto_previous_response_id=auto_previous_response_id,
            conversation_id=conversation_id,
            session=session,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with OpenAI Agents config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["openai_agents"] = OpenAIAgentsExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
