from __future__ import annotations

from agentbridge import dependency_versions


def test_dependency_versions_include_adopted_ranges() -> None:
    versions = dependency_versions()

    assert versions["langgraph"]["adopted_range"] == ">=1.2.11,<2"
    assert versions["pydantic_ai"]["package"] == "pydantic-ai-slim"
    assert versions["pydantic_ai"]["status"] == "verified"
    assert versions["crewai"]["package"] == "agentbridge-crewai"
    assert versions["crewai"]["status"] == "blocked"
    assert "installed_version" in versions["crewai"]
    assert versions["strands"]["package"] == "strands-agents"
    assert versions["strands"]["adopted_range"] == ">=1.55,<2"
    assert versions["strands"]["status"] == "partial"
    assert versions["openai_agents"]["package"] == "openai-agents"
    assert versions["openai_agents"]["adopted_range"] == ">=0.20,<0.21"
    assert versions["google_adk"]["package"] == "google-adk"
    assert versions["google_adk"]["status"] == "partial"
    assert versions["langchain"]["adopted_range"] == ">=1.4,<2"
    assert versions["langchain"]["status"] == "partial"
