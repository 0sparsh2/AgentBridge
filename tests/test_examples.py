from __future__ import annotations

from examples.crewai_prototype_report import build_report as build_crewai_report
from examples.deep_scenario_report import build_report
from examples.google_adk_enterprise_report import build_report as build_google_adk_report
from examples.model_routes import build_model_route_catalog
from examples.openai_agents_approval_report import build_report as build_openai_agents_report
from examples.pydantic_validation_report import build_report as build_pydantic_report
from examples.rag_migration_report import build_report as build_rag_report
from examples.scenario_report_suite import build_suite
from examples.strands_agentcore_report import build_report as build_strands_report


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


def test_rag_migration_report_documents_langchain_to_langgraph_shape() -> None:
    report = build_rag_report()

    assert report["source_framework"] == "langchain"
    assert report["target_framework"] == "langgraph"
    assert report["framework_extensions"]["langchain_source"]["retrievers"] == [
        "refund_policy_docs"
    ]
    assert report["framework_extensions"]["langchain_source"]["store"] == "native_vector_store"
    assert report["framework_extensions"]["langgraph_target"]["enable_checkpointing"] is True
    assert report["model_routes"]["local_ollama"] == "ollama/llama3.1"
    assert report["offline_run_comparison"]["mock"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert "workflow" in report["offline_run_comparison"]["langgraph"]["events"]
    assert "retriever" in " ".join(report["migration_notes"]).lower()


def test_strands_agentcore_report_documents_production_path_shape() -> None:
    report = build_strands_report()

    assert report["source_framework"] == "strands"
    assert report["target_framework"] == "langgraph"
    strands_config = report["framework_extensions"]["strands_source"]
    assert strands_config["deployment_target"] == "agentcore"
    assert strands_config["deployment"]["runtime"] == "bedrock-agentcore"
    assert strands_config["mcp_clients"] == ["orders_mcp", "payments_mcp"]
    assert "refund_policy_guardrail" in strands_config["guardrails"]
    assert "human_review_for_high_value_refunds" in strands_config["interventions"]
    assert report["framework_extensions"]["langgraph_target"]["enable_checkpointing"] is True
    assert report["model_routes"]["local_ollama"] == "ollama/llama3.1"
    assert report["model_routes"]["nvidia_nim_openai_compatible"] == "openai/nvidia-model-name"
    assert report["offline_run_comparison"]["mock"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert "strands" in report["offline_run_comparison"]
    if report["offline_run_comparison"]["strands"]["available"]:
        assert "complete" in report["offline_run_comparison"]["strands"]["events"]
    else:
        assert "error" in report["offline_run_comparison"]["strands"]


def test_google_adk_enterprise_report_documents_service_and_eval_shape() -> None:
    report = build_google_adk_report()

    assert report["source_framework"] == "google_adk"
    assert report["target_framework"] == "langgraph"
    google_config = report["framework_extensions"]["google_adk_source"]
    assert google_config["app_name"] == "enterprise_support"
    assert google_config["session_service"] == "in_memory"
    assert google_config["memory_service"] == "vertex_ai_memory_bank"
    assert google_config["artifact_service"] == "gcs_artifact_service"
    assert google_config["credential_service"] == "secret_manager_credentials"
    assert google_config["sub_agents"] == ["billing_specialist", "technical_specialist"]
    assert google_config["evals"] == ["enterprise_resolution_quality", "policy_compliance"]
    assert google_config["capture_service_snapshots"] is True
    assert google_config["deployment_target"] == "vertex_ai_agent_engine"
    assert report["framework_extensions"]["langgraph_target"]["enable_checkpointing"] is True
    assert report["model_routes"]["google_api"] == "google/gemini"
    assert report["model_routes"]["local_ollama"] == "ollama/llama3.1"
    assert report["offline_run_comparison"]["mock"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert "google_adk" in report["offline_run_comparison"]
    if report["offline_run_comparison"]["google_adk"]["available"]:
        assert "complete" in report["offline_run_comparison"]["google_adk"]["events"]
    else:
        assert "error" in report["offline_run_comparison"]["google_adk"]


def test_pydantic_validation_report_documents_typed_output_shape() -> None:
    report = build_pydantic_report()

    assert report["source_framework"] == "pydantic_ai"
    assert report["target_framework"] == "langgraph"
    pydantic_config = report["framework_extensions"]["pydantic_ai_source"]
    assert pydantic_config["retries"] == 3
    assert pydantic_config["tool_timeout"] == 8
    assert pydantic_config["metadata"]["validation_policy"] == "strict_refund_decision_contract"
    assert report["framework_extensions"]["langgraph_target"]["enable_checkpointing"] is True
    assert report["model_routes"]["offline_test_model"] == "test"
    assert report["model_routes"]["openrouter"] == "openrouter/openai/gpt-4o-mini"
    assert report["capability_deltas"]["typed_output"] == "full"
    assert report["offline_run_comparison"]["mock"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["output"]["eligible"] is True
    assert "pydantic_ai" in report["offline_run_comparison"]
    if report["offline_run_comparison"]["pydantic_ai"]["available"]:
        assert report["offline_run_comparison"]["pydantic_ai"]["output"]["eligible"] is True
        assert "complete" in report["offline_run_comparison"]["pydantic_ai"]["events"]
    else:
        assert "error" in report["offline_run_comparison"]["pydantic_ai"]


def test_openai_agents_approval_report_documents_queue_and_resume_shape() -> None:
    report = build_openai_agents_report()

    assert report["source_framework"] == "openai_agents"
    assert report["target_framework"] == "langgraph"
    openai_config = report["framework_extensions"]["openai_agents_source"]
    assert openai_config["approval_policy"] == {"issue_refund": "required"}
    assert openai_config["approval_store"] == "ApprovalQueue"
    assert openai_config["handoffs"] == ["billing_specialist_agent"]
    assert openai_config["mcp_servers"] == ["orders_mcp"]
    assert openai_config["tracing"] is True
    assert report["framework_extensions"]["langgraph_target"]["enable_checkpointing"] is True
    resume_payload = report["approval_queue_fixture"]["resume_payload"]
    assert resume_payload["id"] == "approval-refund-A123"
    assert resume_payload["backend"] == "openai_agents"
    assert resume_payload["status"] == "approved"
    assert resume_payload["decision"]["reviewer"] == "support_manager"
    assert report["approval_queue_fixture"]["pending_after_decision"] == []
    assert report["offline_run_comparison"]["mock"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert "openai_agents" in report["offline_run_comparison"]
    if report["offline_run_comparison"]["openai_agents"]["available"]:
        assert "complete" in report["offline_run_comparison"]["openai_agents"]["events"]
    else:
        assert "error" in report["offline_run_comparison"]["openai_agents"]


def test_crewai_prototype_report_documents_role_task_migration_shape() -> None:
    report = build_crewai_report()

    assert report["source_framework"] == "crewai"
    assert report["target_framework"] == "langgraph"
    crew_config = report["framework_extensions"]["crewai_source"]
    assert crew_config["role"] == "Refund specialist"
    assert crew_config["process"] == "hierarchical"
    assert crew_config["allow_delegation"] is True
    assert crew_config["memory"] is True
    assert crew_config["human_input"] is True
    assert report["framework_extensions"]["langgraph_target"]["enable_checkpointing"] is True
    assert report["model_routes"]["local_ollama"] == "ollama/llama3.1"
    assert report["capability_deltas"]["roles_tasks_crews"] == "extension"
    assert report["offline_run_comparison"]["mock"]["available"] is True
    assert report["offline_run_comparison"]["langgraph"]["available"] is True
    assert "crewai" in report["offline_run_comparison"]
    if report["offline_run_comparison"]["crewai"]["available"]:
        assert "complete" in report["offline_run_comparison"]["crewai"]["events"]
    else:
        assert "error" in report["offline_run_comparison"]["crewai"]


def test_scenario_report_suite_indexes_all_deep_reports() -> None:
    suite = build_suite()

    assert suite["reports_count"] == 6
    assert suite["source_frameworks"] == [
        "crewai",
        "google_adk",
        "langchain",
        "openai_agents",
        "pydantic_ai",
        "strands",
    ]
    assert suite["target_frameworks"] == ["langgraph"]
    assert set(suite["scenario_index"]) == {
        "crewai_prototype",
        "google_adk_enterprise",
        "openai_agents_approval",
        "pydantic_validation",
        "rag_migration",
        "strands_agentcore",
    }
    assert suite["backend_availability"]["mock"]["available"] == 6
    assert suite["backend_availability"]["langgraph"]["available"] == 6
    assert "crewai" in suite["backend_availability"]
