"""Generate a no-key Strands-to-LangGraph AgentCore migration report."""

from __future__ import annotations

import json

from agentbridge import AgentSpec, ToolSpec, get_adapter, run_agent
from agentbridge.extensions.langgraph import LangGraphExtension
from agentbridge.extensions.strands import StrandsExtension
from examples.model_routes import MODEL_ROUTES


def check_claim(order_id: str) -> str:
    """Return production support context for a refund claim."""

    return f"Order {order_id} has a verified double-charge signal."


agentcore_agent = AgentSpec(
    name="agentcore_refund_agent",
    instructions=(
        "Investigate a refund claim, consult production tools, apply guardrails, "
        "and return an auditable support decision."
    ),
    model="agentbridge/offline",
    tools=[ToolSpec.from_function(check_claim)],
    metadata={
        "owner": "support-platform",
        "scenario": "strands_agentcore_migration",
    },
)


def build_report() -> dict[str, object]:
    strands_source = StrandsExtension.with_config(
        agentcore_agent,
        agent_id="refund-agentcore-v1",
        description="AWS production-path refund support agent.",
        mcp_clients=["orders_mcp", "payments_mcp"],
        guardrails=["refund_policy_guardrail", "pii_redaction_guardrail"],
        hooks=["before_model_call", "after_tool_result"],
        interventions=["human_review_for_high_value_refunds"],
        trace_attributes={
            "service": "refunds",
            "environment": "production",
            "tenant": "retail-demo",
        },
        checkpointing=True,
        session_manager="native_session_manager",
        memory_manager="native_memory_manager",
        structured_output_prompt="Return decision, reason, evidence, and next_action.",
        deployment_target="agentcore",
        deployment={
            "runtime": "bedrock-agentcore",
            "region": "us-east-1",
            "identity": "agentcore-runtime-role",
            "observability": "cloudwatch-and-xray",
        },
        metadata={
            "migration_source": "strands_agentcore",
            "native_features_policy": "preserve via extension config first",
        },
    )
    langgraph_target = LangGraphExtension.with_config(
        agentcore_agent,
        graph_name="refund_agentcore_graph",
        node_name="investigate_refund",
        include_context_in_output=True,
        enable_checkpointing=True,
        route_on_context_key="risk",
        routes={
            "standard": "investigate_refund",
            "high": "human_review",
        },
        interrupt_before=["human_review"],
    )
    return {
        "scenario": "AWS production-path migration",
        "source_framework": "strands",
        "target_framework": "langgraph",
        "agent_spec": agentcore_agent.model_dump(mode="json"),
        "framework_extensions": {
            "strands_source": _json_safe(strands_source.backend_config["strands"]),
            "langgraph_target": langgraph_target.backend_config["langgraph"],
        },
        "model_routes": {
            "offline_ci": MODEL_ROUTES["offline_ci"]["model"],
            "openai_api": MODEL_ROUTES["openai_api"]["model"],
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
            agentcore_agent,
            ["mock", "langgraph", "strands"],
        ),
        "capability_deltas": {
            "tools.sync": "full",
            "mcp_clients": "extension",
            "guardrails": "extension",
            "hooks": "extension",
            "interventions": "extension",
            "agentcore_deployment": "native_only",
            "checkpoint_resume": "partial",
            "model_provider_routing": "partial",
        },
        "migration_notes": [
            "Strands MCP clients, guardrails, hooks, interventions, session/memory managers, and AgentCore deployment settings stay in StrandsExtension config.",
            "LangGraph receives the portable agent contract plus graph routing, checkpointing, and interrupt intent for production orchestration.",
            "App-facing code should compare normalized RunResult output/events while treating native AgentCore deployment execution as a credentialed smoke path.",
            "Hosted APIs, local models, OpenRouter, NVIDIA NIM, and internal OpenAI-compatible gateways are documented as model routes, not reimplemented as a custom provider layer.",
        ],
        "validation": {
            "default": "python examples/strands_agentcore_report.py",
            "credentialed": [
                "Install the external Strands plugin and strands-agents package.",
                "Configure AWS/Bedrock/AgentCore credentials for live deployment smoke tests.",
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
                input="Investigate whether order A123 should be refunded.",
                session_id="strands-agentcore-report",
                context={
                    "order_id": "A123",
                    "risk": "standard",
                    "deployment_path": "agentcore",
                },
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
