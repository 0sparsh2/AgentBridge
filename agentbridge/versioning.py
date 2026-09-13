"""Dependency version reporting."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

ADOPTED_DEPENDENCIES = {
    "pydantic": {
        "package": "pydantic",
        "range": ">=2.13,<3",
        "role": "Core SDK models",
        "status": "verified",
        "notes": "Verified in the local test environment.",
    },
    "litellm": {
        "package": "litellm",
        "range": ">=1.100,<2",
        "role": "Model-provider delegation",
        "status": "verified",
        "notes": "Verified install in the local test environment.",
    },
    "langgraph": {
        "package": "langgraph",
        "range": ">=1.2.11,<2",
        "role": "Stateful graph backend",
        "status": "verified",
        "notes": "Verified with graph execution and ToolSpec calls.",
    },
    "pydantic_ai": {
        "package": "pydantic-ai-slim",
        "range": ">=2.42,<3",
        "role": "Typed-agent backend",
        "status": "verified",
        "notes": "Verified with Pydantic AI TestModel and ToolSpec calls.",
    },
    "crewai": {
        "package": "agentbridge-crewai",
        "range": ">=0.11.2,<0.12",
        "role": "Role/task/crew backend",
        "status": "blocked",
        "notes": (
            "External plugin scaffold exists in plugins/agentbridge-crewai. Verification is blocked "
            "on Python 3.14 because upstream CrewAI 0.11.2 depends on older LangChain/LangSmith "
            "ranges with conflicts in this environment."
        ),
    },
    "openai_agents": {
        "package": "openai-agents",
        "range": ">=0.20,<0.21",
        "role": "OpenAI Agents SDK adapter plugin",
        "status": "partial",
        "notes": (
            "PyPI latest observed as 0.22.2 on 2026-09-11, but 0.22.x requires openai>=3. "
            "The executable baseline is 0.20.x because it remains compatible with the current "
            "LiteLLM/OpenAI dependency path. External plugin maps Agent, Runner, tools, native "
            "structured output, handoff/guardrail/MCP/run options, approval interruption "
            "summaries, guardrail diagnostics, best-effort events, and extension metadata; "
            "application-owned approval UIs/resume stores remain extension-level."
        ),
    },
    "google_adk": {
        "package": "google-adk",
        "range": ">=2.9,<3",
        "role": "Google ADK adapter plugin",
        "status": "partial",
        "notes": (
            "Verified install with google-adk 2.9.0. External plugin maps AgentSpec to ADK "
            "Agent, ToolSpec to FunctionTool, structured output through output_schema plus "
            "typed validation, native Agent/Runner option forwarding, and extension metadata "
            "for sessions, memory, artifacts, evals, deployment, and run diagnostics for "
            "sessions/services/event history/delegation. "
            "Native ADK currently pins older OpenTelemetry packages than Strands, so installing "
            "both native plugins may require isolated environments."
        ),
    },
    "strands": {
        "package": "strands-agents",
        "range": ">=1.55,<2",
        "role": "Strands Agents adapter plugin",
        "status": "partial",
        "notes": (
            "Verified install with strands-agents 1.55.1. External plugin maps AgentSpec "
            "to Strands Agent, ToolSpec to @tool wrappers, structured output model pass-through, "
            "native Agent option forwarding, MCP client/tool-provider pass-through, trace metadata "
            "summaries, structured deployment metadata summaries, and best-effort stream event "
            "normalization. Guardrail labels and live deployment execution remain extension-level."
        ),
    },
    "langchain": {
        "package": "langchain",
        "range": ">=1.4,<2",
        "role": "Direct LangChain adapter plugin",
        "status": "partial",
        "notes": (
            "Verified install with langchain 1.4.0. This complements the built-in LangGraph "
            "backend with direct create_agent execution, StructuredTool mapping, model string "
            "normalization, structured output via native response_format, native create_agent "
            "option forwarding, callback/runtime metadata, stream_events v3 normalization, and "
            "stream_mode v2 normalization. "
            "Portable memory/retriever semantics and LangSmith provider behavior remain "
            "extension-level."
        ),
    },
}


def dependency_versions() -> dict[str, dict[str, str | None]]:
    """Return adopted and installed dependency versions."""

    results: dict[str, dict[str, str | None]] = {}
    for name, dependency in ADOPTED_DEPENDENCIES.items():
        package = dependency["package"]
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        results[name] = {
            "package": package,
            "adopted_range": dependency["range"],
            "installed_version": installed,
            "role": dependency["role"],
            "status": dependency["status"],
            "notes": dependency["notes"],
        }
    return results
