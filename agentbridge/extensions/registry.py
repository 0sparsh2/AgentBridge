"""Extension namespace registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.crewai import CrewAIConfig
from agentbridge.extensions.google_adk import GoogleADKConfig
from agentbridge.extensions.langchain import LangChainConfig
from agentbridge.extensions.langgraph import LangGraphConfig
from agentbridge.extensions.openai_agents import OpenAIAgentsConfig
from agentbridge.extensions.pydantic_ai import PydanticAIConfig
from agentbridge.extensions.strands import StrandsConfig


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
        ExtensionProfile(
            framework="openai_agents",
            module="agentbridge.extensions.openai_agents",
            config_model="OpenAIAgentsConfig",
            config_schema=OpenAIAgentsConfig.model_json_schema(),
            capabilities=[
                "workflow.handoffs",
                "guardrails",
                "human_approval",
                "observability.tracing",
            ],
            status="partial",
            notes=[
                "External plugin maps AgentSpec and ToolSpec to OpenAI Agents SDK Agent/Runner on the compatible 0.20.x line.",
                "Handoffs, guardrails, tracing, approvals, and latest 0.22.x verification remain blocked or extension-level until dependency conflicts are resolved.",
            ],
        ),
        ExtensionProfile(
            framework="google_adk",
            module="agentbridge.extensions.google_adk",
            config_model="GoogleADKConfig",
            config_schema=GoogleADKConfig.model_json_schema(),
            capabilities=[
                "state.session",
                "state.memory",
                "workflow.delegation",
                "deployment.serverless",
                "evals",
            ],
            status="scaffolded",
            notes=[
                "External plugin target for Google ADK sessions, memory services, sub-agents, evals, and deployment.",
                "Native service wiring should remain extension-level until portable semantics are proven.",
            ],
        ),
        ExtensionProfile(
            framework="strands",
            module="agentbridge.extensions.strands",
            config_model="StrandsConfig",
            config_schema=StrandsConfig.model_json_schema(),
            capabilities=[
                "tools.mcp",
                "structured_output",
                "streaming.events",
                "guardrails",
                "observability.tracing",
                "deployment.serverless",
            ],
            status="scaffolded",
            notes=[
                "External plugin target for Strands tools, MCP clients, hooks, structured output, and AWS deployment paths.",
                "Hook-driven lifecycle support should drive future AgentEvent expansion.",
            ],
        ),
        ExtensionProfile(
            framework="langchain",
            module="agentbridge.extensions.langchain",
            config_model="LangChainConfig",
            config_schema=LangChainConfig.model_json_schema(),
            capabilities=[
                "tools.sync",
                "tools.async",
                "state.memory",
                "observability.tracing",
            ],
            status="scaffolded",
            notes=[
                "Complements the built-in LangGraph adapter with direct LangChain agent, middleware, callback, memory, and retriever configuration.",
                "LangGraph remains the preferred stateful orchestration backend; this plugin targets broader LangChain app compatibility.",
            ],
        ),
    ]


def extension_profile(framework: str) -> ExtensionProfile:
    """Return one extension profile by framework name."""

    for profile in extension_profiles():
        if profile.framework == framework:
            return profile
    raise ValueError(f"Unknown extension framework: {framework}")
