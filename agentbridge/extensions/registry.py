"""Extension namespace registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.crewai import CrewAIConfig
from agentbridge.extensions.langgraph import LangGraphConfig
from agentbridge.extensions.pydantic_ai import PydanticAIConfig


class ExtensionProfile(BaseModel):
    """Discoverable metadata for a framework extension namespace."""

    framework: str
    module: str
    config_model: str
    config_schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    status: str
    notes: list[str] = Field(default_factory=list)


def extension_profiles() -> list[ExtensionProfile]:
    """Return metadata for built-in extension namespaces."""

    return [
        ExtensionProfile(
            framework="langgraph",
            module="agentbridge.extensions.langgraph",
            config_model="LangGraphConfig",
            config_schema=LangGraphConfig.model_json_schema(),
            capabilities=[
                "workflow.graph",
                "state.checkpointing",
                "human_approval",
            ],
            status="partial",
            notes=[
                "Supports node naming, graph naming, context echoing, and in-memory checkpointing.",
                "Conditional routing and human interrupt helpers are planned.",
            ],
        ),
        ExtensionProfile(
            framework="pydantic_ai",
            module="agentbridge.extensions.pydantic_ai",
            config_model="PydanticAIConfig",
            config_schema=PydanticAIConfig.model_json_schema(),
            capabilities=[
                "structured_output",
                "structured_output.validation_retries",
            ],
            status="partial",
            notes=[
                "Supports retries, tool timeout, metadata, and offline TestModel output controls.",
                "Dependency injection helpers are planned.",
            ],
        ),
        ExtensionProfile(
            framework="crewai",
            module="agentbridge.extensions.crewai",
            config_model="CrewAIConfig",
            config_schema=CrewAIConfig.model_json_schema(),
            capabilities=[
                "workflow.roles_tasks",
                "workflow.delegation",
                "memory.long_term",
                "human_approval",
            ],
            status="scaffolded",
            notes=[
                "External plugin maps role, goal, task, process, delegation, memory, and human input.",
                "Real CrewAI execution remains blocked in this Python 3.14 workspace.",
            ],
        ),
    ]


def extension_profile(framework: str) -> ExtensionProfile:
    """Return one extension profile by framework name."""

    for profile in extension_profiles():
        if profile.framework == framework:
            return profile
    raise ValueError(f"Unknown extension framework: {framework}")
