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
        "range": ">=0.22,<1",
        "role": "OpenAI Agents SDK adapter plugin",
        "status": "planned",
        "notes": (
            "PyPI latest observed as 0.22.2 on 2026-09-11. External plugin scaffold "
            "targets Agent, Runner, tools, handoffs, guardrails, approvals, and tracing."
        ),
    },
    "google_adk": {
        "package": "google-adk",
        "range": ">=2.9,<3",
        "role": "Google ADK adapter plugin",
        "status": "planned",
        "notes": (
            "PyPI latest observed as 2.9.0 on 2026-09-11. External plugin scaffold "
            "targets sessions, memory services, sub-agents, evals, and deployment surfaces."
        ),
    },
    "strands": {
        "package": "strands-agents",
        "range": ">=1.55,<2",
        "role": "Strands Agents adapter plugin",
        "status": "planned",
        "notes": (
            "PyPI latest observed as 1.55.1 on 2026-09-11. External plugin scaffold "
            "targets Agent/tool mapping, MCP clients, hooks, structured output, and tracing."
        ),
    },
    "langchain": {
        "package": "langchain",
        "range": ">=1.4,<2",
        "role": "Direct LangChain adapter plugin",
        "status": "planned",
        "notes": (
            "PyPI latest observed as 1.4.0 on 2026-09-11. This complements the built-in "
            "LangGraph backend with broader LangChain agent, middleware, callback, memory, "
            "and retriever compatibility."
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
