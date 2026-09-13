"""Generate a no-key deep scenario report for AgentBridge planning.

This example is intentionally inspectable without cloud credentials. It shows how a report can
capture framework nuance, migration direction, normalized output expectations, and model routes.
"""

from __future__ import annotations

import json

from agentbridge import AgentSpec, ApprovalQueue, get_adapter, run_agent
from agentbridge.extensions.google_adk import GoogleADKExtension
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge.extensions.langgraph import LangGraphExtension
from agentbridge.extensions.openai_agents import OpenAIAgentsExtension
from agentbridge.extensions.strands import StrandsExtension
from examples.model_routes import MODEL_ROUTES


base_agent = AgentSpec(
    name="refund_approval_agent",
    instructions="Check refund eligibility, request approval when needed, and return a decision.",
    model="agentbridge/offline",
)


def build_report() -> dict[str, object]:
    approval_queue = ApprovalQueue()
    variants = {
        "openai_agents_source": OpenAIAgentsExtension.with_config(
            base_agent,
            approval_policy={"issue_refund": "required"},
            approval_store=approval_queue,
            handoff_description="Escalate payment disputes to billing.",
            tracing=True,
        ),
        "langgraph_target": LangGraphExtension.with_config(
            base_agent,
            graph_name="refund_approval_graph",
            node_name="refund_decision",
            enable_checkpointing=True,
            interrupt_before=["refund_decision"],
            route_on_context_key="intent",
            routes={"refund": "refund_decision", "billing": "billing_review"},
        ),
        "langchain_rag_variant": LangChainExtension.with_config(
            base_agent,
            retrievers=["refund_policy_docs"],
            memory="conversation_buffer",
            callbacks=["langsmith"],
            checkpointer="native_checkpointer",
            store="native_vector_store",
        ),
        "strands_aws_variant": StrandsExtension.with_config(
            base_agent,
            mcp_clients=["orders_mcp"],
            guardrails=["refund_policy"],
            trace_attributes={"service": "refunds"},
            deployment_target="agentcore",
            deployment={"runtime": "bedrock-agentcore", "region": "us-east-1"},
        ),
        "google_adk_enterprise_variant": GoogleADKExtension.with_config(
            base_agent,
            app_name="refund_support",
            session_service="in_memory",
            memory_service="vertex",
            artifact_service="gcs",
            sub_agents=["billing_agent"],
            evals=["refund_quality_eval"],
            capture_service_snapshots=True,
            deployment_target="vertex_ai",
            deployment={"runtime": "adk", "region": "us-central1"},
        ),
    }
    return {
        "scenario": "refund approval migration",
        "source_framework": "openai_agents",
        "target_framework": "langgraph",
        "agent_spec": base_agent.model_dump(mode="json"),
        "framework_extensions": {
            name: _json_safe(next(iter(agent.backend_config.values())))
            for name, agent in variants.items()
        },
        "model_routes": {
            name: route["model"]
            for name, route in MODEL_ROUTES.items()
        },
        "expected_normalized_output": {
            "run_result": ["output", "backend", "events", "usage", "metadata", "raw"],
            "events": ["message", "tool_call", "tool_result", "workflow", "complete"],
            "migration_notes": [
                "OpenAI approval interruptions become ApprovalQueue records.",
                "LangGraph target should preserve approval/resume semantics as checkpoint interrupts.",
                "RAG, MCP, ADK services, and deployment metadata remain extension-specific.",
            ],
        },
        "offline_run_comparison": _offline_run_comparison(
            base_agent,
            ["mock", "langgraph", "langchain", "openai_agents", "google_adk", "strands"],
        ),
        "default_mode": "No-key metadata/report example. Use credentialed runs only in separate smoke tests.",
    }


def _offline_run_comparison(agent: AgentSpec, backends: list[str]) -> dict[str, object]:
    comparison: dict[str, object] = {}
    for backend in backends:
        try:
            get_adapter(backend)
            result = run_agent(
                agent,
                backend=backend,
                input="Customer says order A123 was double charged.",
                session_id="scenario-report",
                context={"intent": "refund"},
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
