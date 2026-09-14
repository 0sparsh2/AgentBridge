"""Generate a no-key OpenAI Agents-to-LangGraph approval migration report."""

from __future__ import annotations

import json

from agentbridge import AgentSpec, ApprovalQueue, ToolSpec, get_adapter, run_agent
from agentbridge.extensions.langgraph import LangGraphExtension
from agentbridge.extensions.openai_agents import OpenAIAgentsExtension
from examples.model_routes import MODEL_ROUTES


def issue_refund(order_id: str) -> str:
    """Issue a refund after human approval."""

    return f"Refund issued for order {order_id}."


approval_agent = AgentSpec(
    name="approval_refund_agent",
    instructions=(
        "Investigate refund eligibility, pause for human approval before issuing "
        "a refund, and preserve enough run state for resume."
    ),
    model="agentbridge/offline",
    tools=[ToolSpec.from_function(issue_refund)],
    metadata={
        "owner": "support-ops",
        "scenario": "openai_agents_approval_migration",
    },
)


def build_report() -> dict[str, object]:
    approval_queue = ApprovalQueue()
    approval_fixture = _approval_fixture(approval_queue)
    openai_source = OpenAIAgentsExtension.with_config(
        approval_agent,
        approval_policy={"issue_refund": "required"},
        approval_store=approval_queue,
        handoff_description="Escalate disputed billing cases to a billing specialist.",
        handoffs=["billing_specialist_agent"],
        mcp_servers=["orders_mcp"],
        guardrails=["refund_policy_guardrail"],
        input_guardrails=["pii_redaction_guardrail"],
        output_guardrails=["no_sensitive_refund_data"],
        tool_use_behavior="stop_on_first_tool",
        tracing=True,
        max_turns=5,
        previous_response_id="resp_previous_refund_check",
        auto_previous_response_id=False,
        conversation_id="conv_refund_approval",
        metadata={
            "migration_source": "openai_agents_approval",
            "native_features_policy": "store interruptions and resume payloads",
        },
    )
    langgraph_target = LangGraphExtension.with_config(
        approval_agent,
        graph_name="approval_refund_graph",
        node_name="refund_decision",
        include_context_in_output=True,
        enable_checkpointing=True,
        route_on_context_key="approval_state",
        routes={
            "approved": "refund_decision",
            "needs_approval": "human_review",
        },
        interrupt_before=["human_review"],
    )
    return {
        "scenario": "refund approval migration",
        "source_framework": "openai_agents",
        "target_framework": "langgraph",
        "agent_spec": approval_agent.model_dump(mode="json"),
        "framework_extensions": {
            "openai_agents_source": _json_safe(
                openai_source.backend_config["openai_agents"]
            ),
            "langgraph_target": langgraph_target.backend_config["langgraph"],
        },
        "approval_queue_fixture": approval_fixture,
        "model_routes": {
            "offline_ci": MODEL_ROUTES["offline_ci"]["model"],
            "openai_api": MODEL_ROUTES["openai_api"]["model"],
            "openrouter": MODEL_ROUTES["openrouter"]["model"],
            "nvidia_nim_openai_compatible": MODEL_ROUTES[
                "nvidia_nim_openai_compatible"
            ]["model"],
            "custom_openai_compatible_gateway": MODEL_ROUTES[
                "custom_openai_compatible_gateway"
            ]["model"],
        },
        "offline_run_comparison": _offline_run_comparison(
            approval_agent,
            ["mock", "openai_agents", "langgraph"],
        ),
        "capability_deltas": {
            "handoffs": "extension",
            "mcp_servers": "extension",
            "guardrails": "extension",
            "human_approval": "extension",
            "approval_queue": "full",
            "native_resume_execution": "native_only",
            "langgraph_interrupt_resume": "partial",
            "tracing": "extension",
            "model_provider_routing": "partial",
        },
        "migration_notes": [
            "OpenAI Agents approval interruptions are normalized into workflow events and can be persisted into app-owned stores or ApprovalQueue.",
            "ApprovalQueue produces a backend-neutral resume payload, while native SDK resume execution remains backend-specific.",
            "LangGraph maps the approval boundary to checkpointed interrupts and resume intent.",
            "Handoffs, MCP servers, guardrails, tracing, response IDs, and runner options remain OpenAI Agents extension settings.",
            "No-key output comparison runs the portable base spec; native handoff/guardrail objects should be supplied in credentialed SDK smoke tests.",
            "OpenAI, OpenRouter, NVIDIA NIM, and internal OpenAI-compatible gateways stay documented as model routes without creating a second provider abstraction.",
        ],
        "validation": {
            "default": "python examples/openai_agents_approval_report.py",
            "credentialed": [
                "Install the external OpenAI Agents plugin and compatible openai-agents package.",
                "Configure provider credentials for live SDK approval interruption and native resume smoke tests.",
                "Configure provider-specific API keys or base URLs for OpenRouter, NVIDIA NIM, or custom gateway routes.",
            ],
        },
    }


def _approval_fixture(queue: ApprovalQueue) -> dict[str, object]:
    queue.record_approval_request(
        {
            "id": "approval-refund-A123",
            "backend": "openai_agents",
            "status": "pending",
            "interruption": {
                "tool_name": "issue_refund",
                "arguments": {"order_id": "A123"},
                "reason": "refund tool requires human approval",
            },
            "state": {"conversation_id": "conv_refund_approval"},
            "state_type": "OpenAIAgentsRunState",
            "last_response_id": "resp_pending_approval",
            "last_agent": {"name": "approval_refund_agent"},
            "metadata": {"source": "scenario_report_fixture"},
        }
    )
    resume_payload = queue.approve(
        "approval-refund-A123",
        response={"approved": True, "comment": "Duplicate charge verified."},
        reviewer="support_manager",
        metadata={"review_channel": "agentbridge_report"},
    )
    return {
        "pending_after_decision": queue.pending(backend="openai_agents"),
        "resume_payload": _json_safe(resume_payload),
    }


def _offline_run_comparison(agent: AgentSpec, backends: list[str]) -> dict[str, object]:
    comparison: dict[str, object] = {}
    for backend in backends:
        try:
            get_adapter(backend)
            result = run_agent(
                agent,
                backend=backend,
                input="Order A123 has a verified duplicate charge. Continue refund.",
                session_id="openai-agents-approval-report",
                context={
                    "order_id": "A123",
                    "approval_state": "approved",
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
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return type(value).__name__


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
