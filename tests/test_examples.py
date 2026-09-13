from __future__ import annotations

from examples.deep_scenario_report import build_report
from examples.model_routes import build_model_route_catalog


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
    assert report["offline_run_comparison"]["mock"] == {
        "available": True,
        "backend": "mock",
        "output": {
            "agent": "refund_approval_agent",
            "input": "Customer says order A123 was double charged.",
            "message": "Mock backend completed successfully.",
            "tools": [],
        },
        "events": ["message", "complete"],
        "metadata_keys": [],
    }
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["backend"] == "langgraph"
    assert "openai_agents" in report["offline_run_comparison"]
    if report["offline_run_comparison"]["openai_agents"]["available"]:
        assert "complete" in report["offline_run_comparison"]["openai_agents"]["events"]
    else:
        assert "error" in report["offline_run_comparison"]["openai_agents"]


def test_model_route_catalog_documents_optional_provider_shapes() -> None:
    catalog = build_model_route_catalog()
    routes = catalog["routes"]

    assert routes["offline_ci"]["default_ci"] is True
    assert routes["offline_ci"]["credentialed"] is False
    assert routes["local_ollama"]["model"] == "ollama/llama3.1"
    assert routes["openrouter"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert routes["nvidia_nim_openai_compatible"]["base_url_env"] == "NVIDIA_NIM_BASE_URL"
    assert routes["custom_openai_compatible_gateway"]["model"] == "openai/internal-agent-model"
