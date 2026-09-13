from __future__ import annotations

from examples.deep_scenario_report import build_report


def test_deep_scenario_report_documents_frameworks_and_model_routes() -> None:
    report = build_report()

    assert report["source_framework"] == "openai_agents"
    assert report["target_framework"] == "langgraph"
    assert set(report["framework_extensions"]) == {
        "google_adk_enterprise_variant",
        "langchain_rag_variant",
        "langgraph_target",
        "openai_agents_source",
        "strands_aws_variant",
    }
    assert report["model_routes"] == {
        "offline_ci": "agentbridge/offline",
        "openai_api": "openai/gpt-5",
        "anthropic_api": "anthropic/claude-sonnet",
        "google_api": "google/gemini",
        "local_ollama": "ollama/llama3.1",
        "openrouter": "openrouter/openai/gpt-4o-mini",
        "nvidia_nim_openai_compatible": "openai/nvidia-model-name",
        "custom_openai_compatible_gateway": "openai/internal-agent-model",
    }
    assert report["framework_extensions"]["openai_agents_source"]["approval_store"] == "ApprovalQueue"
    assert "workflow" in report["expected_normalized_output"]["events"]
