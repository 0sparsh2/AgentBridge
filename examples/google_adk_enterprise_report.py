"""Generate a no-key Google ADK-to-LangGraph enterprise migration report."""

from __future__ import annotations

import json

from agentbridge import AgentSpec, ToolSpec, get_adapter, run_agent
from agentbridge.extensions.google_adk import GoogleADKExtension
from agentbridge.extensions.langgraph import LangGraphExtension
from examples.model_routes import MODEL_ROUTES


def classify_case(case_id: str) -> str:
    """Return enterprise support routing context for a case."""

    return f"Case {case_id} is a priority enterprise account escalation."


enterprise_agent = AgentSpec(
    name="enterprise_support_agent",
    instructions=(
        "Resolve an enterprise support case, preserve session state, capture "
        "artifacts, and escalate to specialist sub-agents when required."
    ),
    model="agentbridge/offline",
    tools=[ToolSpec.from_function(classify_case)],
    metadata={
        "owner": "enterprise-support",
        "scenario": "google_adk_enterprise_migration",
    },
)


def build_report() -> dict[str, object]:
    google_source = GoogleADKExtension.with_config(
        enterprise_agent,
        app_name="enterprise_support",
        description="Google ADK enterprise support agent with service integrations.",
        global_instruction="Follow enterprise support compliance policy.",
        static_instruction="Never expose internal account identifiers.",
        state_schema={
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "tier": {"type": "string"},
            },
        },
        generate_content_config="native_generate_content_config",
        include_contents="default",
        output_key="support_resolution",
        planner="native_planner",
        code_executor="native_code_executor",
        retry_config="native_retry_config",
        timeout=30.0,
        before_agent_callback="audit_before_agent",
        after_agent_callback="audit_after_agent",
        before_model_callback="redact_before_model",
        after_model_callback="trace_after_model",
        before_tool_callback="authorize_tool_use",
        after_tool_callback="capture_tool_result",
        session_service="in_memory",
        memory_service="vertex_ai_memory_bank",
        artifact_service="gcs_artifact_service",
        credential_service="secret_manager_credentials",
        runner_plugins=["adk_observability_plugin"],
        plugin_close_timeout=5.0,
        auto_create_session=True,
        sub_agents=["billing_specialist", "technical_specialist"],
        evals=["enterprise_resolution_quality", "policy_compliance"],
        eval_runner="adk_eval_runner",
        capture_service_snapshots=True,
        deployment_target="vertex_ai_agent_engine",
        deployment={
            "runtime": "google-adk",
            "region": "us-central1",
            "project": "enterprise-support-prod",
            "artifact_store": "gs://agentbridge-enterprise-artifacts",
        },
        metadata={
            "migration_source": "google_adk_enterprise",
            "native_features_policy": "capture service bindings and eval metadata",
        },
    )
    langgraph_target = LangGraphExtension.with_config(
        enterprise_agent,
        graph_name="enterprise_support_graph",
        node_name="triage_case",
        include_context_in_output=True,
        enable_checkpointing=True,
        route_on_context_key="tier",
        routes={
            "standard": "triage_case",
            "enterprise": "specialist_review",
        },
        interrupt_before=["specialist_review"],
    )
    return {
        "scenario": "Google enterprise path migration",
        "source_framework": "google_adk",
        "target_framework": "langgraph",
        "agent_spec": enterprise_agent.model_dump(mode="json"),
        "framework_extensions": {
            "google_adk_source": google_source.backend_config["google_adk"],
            "langgraph_target": langgraph_target.backend_config["langgraph"],
        },
        "model_routes": {
            "offline_ci": MODEL_ROUTES["offline_ci"]["model"],
            "google_api": MODEL_ROUTES["google_api"]["model"],
            "local_ollama": MODEL_ROUTES["local_ollama"]["model"],
            "openrouter": MODEL_ROUTES["openrouter"]["model"],
            "nvidia_nim_openai_compatible": MODEL_ROUTES[
                "nvidia_nim_openai_compatible"
            ]["model"],
            "custom_openai_compatible_gateway": MODEL_ROUTES[
                "custom_openai_compatible_gateway"
            ]["model"],
        },
        "offline_run_comparison": _offline_run_comparison(
            enterprise_agent,
            ["mock", "langgraph", "google_adk"],
        ),
        "capability_deltas": {
            "session_service": "extension",
            "memory_service": "extension",
            "artifact_service": "extension",
            "credential_service": "extension",
            "callbacks": "extension",
            "sub_agents": "extension",
            "eval_execution": "native_only",
            "deployment_publish": "native_only",
            "checkpoint_resume": "partial",
            "model_provider_routing": "partial",
        },
        "migration_notes": [
            "Google ADK session, memory, artifact, credential, plugin, callback, sub-agent, eval, and deployment settings remain extension-specific.",
            "LangGraph receives the portable agent contract plus checkpoint, routing, and interrupt intent for production orchestration.",
            "Service snapshots are captured when native services are available; string labels remain inspectable metadata in no-key reports.",
            "Live eval execution and Vertex/Agent Engine deployment publishing are credentialed smoke paths, not CI defaults.",
            "Google Gemini, local models, OpenRouter, NVIDIA NIM, and internal OpenAI-compatible gateways are documented as model routes without adding a custom provider layer.",
        ],
        "validation": {
            "default": "python examples/google_adk_enterprise_report.py",
            "credentialed": [
                "Install the external Google ADK plugin and google-adk package.",
                "Configure Google Cloud credentials for session, memory, artifact, eval, and deployment smoke tests.",
                "Configure provider-specific API keys or base URLs for hosted/local model routes.",
            ],
        },
    }


def _offline_run_comparison(agent: AgentSpec, backends: list[str]) -> dict[str, object]:
    comparison: dict[str, object] = {}
    for backend in backends:
        try:
            get_adapter(backend)
            result = run_agent(
                agent,
                backend=backend,
                input="Resolve enterprise support case C456.",
                session_id="google-adk-enterprise-report",
                context={
                    "case_id": "C456",
                    "tier": "standard",
                    "deployment_path": "vertex_ai_agent_engine",
                },
                metadata={"user_id": "enterprise-user"},
            )
        except Exception as exc:  # pragma: no cover - depends on optional local plugins.
            comparison[backend] = {
                "available": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
            continue
        comparison[backend] = {
            "available": True,
            "backend": result.backend,
            "output": _json_safe(result.output),
            "events": [event.type for event in result.events],
            "metadata_keys": sorted(result.metadata),
        }
    return comparison


def _json_safe(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return type(value).__name__


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
