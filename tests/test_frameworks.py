from __future__ import annotations

import json

import pytest

from agentbridge.frameworks import framework_support_report, load_framework_manifests


def test_framework_manifests_define_existing_adapter_targets() -> None:
    manifests = {manifest.id: manifest for manifest in load_framework_manifests()}

    assert {
        "agentcore",
        "crewai",
        "deepagents",
        "google_adk",
        "langchain",
        "langgraph",
        "openai_agents",
        "pydantic_ai",
        "strands",
    }.issubset(manifests)
    assert manifests["langgraph"].status == "verified"
    assert "credentialed" in manifests["langchain"].verification_lanes
    assert manifests["crewai"].status == "blocked"


def test_framework_manifest_rejects_unknown_capability(tmp_path) -> None:
    (tmp_path / "invalid.json").write_text(
        json.dumps(
            {
                "id": "invalid",
                "display_name": "Invalid",
                "status": "planned",
                "documentation_url": "https://example.test/docs",
                "capabilities": {"unknown.capability": "full"},
                "next_slice": "Do not use.",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical taxonomy"):
        load_framework_manifests(tmp_path)


def test_framework_support_report_is_renderable() -> None:
    report = framework_support_report()

    assert report.framework_statuses["partial"] >= 1
    assert report.capability_statuses["extension"] >= 1
    markdown = report.as_markdown()
    assert "# Framework Support" in markdown
    assert "## LangChain" in markdown
    assert "`credentialed`" in markdown
