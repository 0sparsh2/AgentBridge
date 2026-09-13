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
                "workflow.routing",
                "state.checkpointing",
                "human_approval",
                "observability.diagnostics",
            ],
            status="partial",
            notes=[
                "Supports node naming, graph naming, context echoing, conditional routing, in-memory checkpointing, and run diagnostics.",
                "Checkpointed interrupts can be resumed with resume_agent() when the same compiled runtime owns the checkpoint state.",
                "Backend-neutral review queues and approval policy models remain extension-level.",
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
                "structured_output",
                "workflow.handoffs",
                "tools.mcp",
                "guardrails",
                "human_approval",
                "observability.tracing",
                "observability.diagnostics",
            ],
            status="partial",
            notes=[
                "External plugin maps AgentSpec and ToolSpec to OpenAI Agents SDK Agent/Runner on the compatible 0.20.x line.",
                "Structured output, native handoff/guardrail/MCP/run options, approval interruption summaries, app-owned approval request stores, guardrail diagnostics, tracing summaries, and best-effort events are covered by the plugin.",
                "Backend-neutral approval UIs/resume queues and latest 0.22.x verification remain blocked or extension-level until dependency conflicts are resolved.",
            ],
        ),
        ExtensionProfile(
            framework="google_adk",
            module="agentbridge.extensions.google_adk",
            config_model="GoogleADKConfig",
            config_schema=GoogleADKConfig.model_json_schema(),
            capabilities=[
                "structured_output",
                "state.session",
                "state.memory",
                "workflow.delegation",
                "deployment.serverless",
                "evals",
                "observability.diagnostics",
            ],
            status="partial",
            notes=[
                "External plugin maps AgentSpec to ADK Agent, ToolSpec to FunctionTool, and runs through Runner with in-memory sessions.",
                "Structured output is validated through ADK output_schema plus AgentBridge typed validation; run diagnostics summarize sessions, services, event history, and transfer-to-agent events.",
                "Native memory/artifact service behavior, eval execution, and deployment publishing remain extension-level.",
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
                "observability.diagnostics",
                "deployment.serverless",
            ],
            status="partial",
            notes=[
                "External plugin maps AgentSpec and ToolSpec to Strands Agent and @tool wrappers.",
                "Structured output, native Agent option forwarding, trace summaries, MCP client/tool-provider object pass-through, hook/intervention/guardrail lifecycle event normalization, run diagnostics, and deployment metadata summaries are covered by the plugin.",
                "Plain string MCP labels, guardrail labels, end-to-end MCP server/client fixtures, and live AWS deployment execution remain extension-level.",
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
                "structured_output",
                "state.memory",
                "observability.tracing",
                "observability.diagnostics",
            ],
            status="partial",
            notes=[
                "External plugin maps AgentSpec to LangChain create_agent and ToolSpec to StructuredTool.",
                "Structured output is validated through native response_format and structured_response; run diagnostics summarize messages, tools, runtime config, and extension options.",
                "LangGraph remains the preferred stateful orchestration backend; callbacks, memory, retrievers, and LangSmith provider behavior remain extension-level.",
            ],
        ),
    ]


def extension_profile(framework: str) -> ExtensionProfile:
    """Return one extension profile by framework name."""

    for profile in extension_profiles():
        if profile.framework == framework:
            return profile
    raise ValueError(f"Unknown extension framework: {framework}")
