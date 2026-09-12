"""Strands Agents-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class StrandsConfig(BaseModel):
    """AgentBridge config for Strands-native behavior."""

    model_config = {"arbitrary_types_allowed": True}

    conversation_manager: Any | None = None
    context_manager: Any | None = None
    hooks: list[Any] = Field(default_factory=list)
    plugins: list[Any] = Field(default_factory=list)
    interventions: list[Any] = Field(default_factory=list)
    mcp_clients: list[Any] = Field(default_factory=list)
    trace_attributes: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[Any] = Field(default_factory=list)
    session_manager: Any | None = None
    memory_manager: Any | None = None
    tool_executor: Any | None = None
    retry_strategy: Any | None = None
    checkpointing: bool | None = None
    sandbox: Any | None = None
    storage: Any | None = None
    background_tasks: Any | None = None
    agent_id: str | None = None
    description: str | None = None
    structured_output_prompt: str | None = None
    load_tools_from_directory: bool | None = None
    record_direct_tool_call: bool | None = None
    deployment_target: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrandsExtension(FrameworkExtension):
    """Extension namespace for Strands Agents behavior."""

    framework = "strands"

    @staticmethod
    def config(
        *,
        conversation_manager: Any | None = None,
        context_manager: Any | None = None,
        hooks: list[Any] | None = None,
        plugins: list[Any] | None = None,
        interventions: list[Any] | None = None,
        mcp_clients: list[Any] | None = None,
        trace_attributes: dict[str, Any] | None = None,
        guardrails: list[Any] | None = None,
        session_manager: Any | None = None,
        memory_manager: Any | None = None,
        tool_executor: Any | None = None,
        retry_strategy: Any | None = None,
        checkpointing: bool | None = None,
        sandbox: Any | None = None,
        storage: Any | None = None,
        background_tasks: Any | None = None,
        agent_id: str | None = None,
        description: str | None = None,
        structured_output_prompt: str | None = None,
        load_tools_from_directory: bool | None = None,
        record_direct_tool_call: bool | None = None,
        deployment_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable Strands adapter configuration."""

        return StrandsConfig(
            conversation_manager=conversation_manager,
            context_manager=context_manager,
            hooks=hooks or [],
            plugins=plugins or [],
            interventions=interventions or [],
            mcp_clients=mcp_clients or [],
            trace_attributes=trace_attributes or {},
            guardrails=guardrails or [],
            session_manager=session_manager,
            memory_manager=memory_manager,
            tool_executor=tool_executor,
            retry_strategy=retry_strategy,
            checkpointing=checkpointing,
            sandbox=sandbox,
            storage=storage,
            background_tasks=background_tasks,
            agent_id=agent_id,
            description=description,
            structured_output_prompt=structured_output_prompt,
            load_tools_from_directory=load_tools_from_directory,
            record_direct_tool_call=record_direct_tool_call,
            deployment_target=deployment_target,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with Strands config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["strands"] = StrandsExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
