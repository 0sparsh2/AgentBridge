"""Generate a no-key CrewAI-to-LangGraph prototype migration report."""

from __future__ import annotations

import json

from agentbridge import AgentSpec, ToolSpec, get_adapter, run_agent
from agentbridge.extensions.crewai import CrewAIExtension
from agentbridge.extensions.langgraph import LangGraphExtension
from examples.model_routes import MODEL_ROUTES


def summarize_policy(order_id: str) -> str:
    """Return refund policy context for a CrewAI-style task."""

    return f"Order {order_id} is covered by duplicate-charge refund policy."


prototype_agent = AgentSpec(
    name="crew_refund_agent",
    instructions="Resolve refund requests using role/task collaboration.",
    model="agentbridge/offline",
    tools=[ToolSpec.from_function(summarize_policy)],
    metadata={
        "owner": "support-prototyping",
        "scenario": "crewai_prototype_migration",
    },
)


def build_report() -> dict[str, object]:
    crew_source = CrewAIExtension.with_config(
        prototype_agent,
        role="Refund specialist",
        goal="Resolve refund requests using support policy and escalation rules.",
        backstory="Expert in billing disputes, refund policy, and customer empathy.",
        task_description=(
            "Review the customer request: {input}. Use policy context and decide whether "
            "to refund, escalate, or request more evidence."
        ),
        expected_output="A refund decision with rationale, owner, and next action.",
        process="hierarchical",
        verbose=True,
        allow_delegation=True,
        memory=True,
        human_input=True,
        metadata={
            "migration_source": "crewai_prototype",
            "native_features_policy": "preserve role/task/crew semantics in extension config",
        },
    )
    langgraph_target = LangGraphExtension.with_config(
        prototype_agent,
        graph_name="crew_refund_production_graph",
        node_name="refund_specialist",
        include_context_in_output=True,
        enable_checkpointing=True,
        route_on_context_key="case_state",
        routes={
            "standard": "refund_specialist",
            "needs_manager": "manager_review",
        },
        interrupt_before=["manager_review"],
    )
    return {
        "scenario": "CrewAI prototype-to-production migration",
        "source_framework": "crewai",
        "target_framework": "langgraph",
        "agent_spec": prototype_agent.model_dump(mode="json"),
        "framework_extensions": {
            "crewai_source": crew_source.backend_config["crewai"],
            "langgraph_target": langgraph_target.backend_config["langgraph"],
        },
        "model_routes": {
            "offline_ci": MODEL_ROUTES["offline_ci"]["model"],
            "openai_api": MODEL_ROUTES["openai_api"]["model"],
            "anthropic_api": MODEL_ROUTES["anthropic_api"]["model"],
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
            prototype_agent,
            ["mock", "langgraph", "crewai"],
        ),
        "capability_deltas": {
            "roles_tasks_crews": "extension",
            "delegation": "extension",
            "memory": "extension",
            "human_input": "extension",
            "native_crewai_execution": "native_only",
            "langgraph_checkpointing": "partial",
            "model_provider_routing": "partial",
        },
        "migration_notes": [
            "CrewAI role, goal, backstory, task, expected output, process, delegation, memory, and human-input settings remain CrewAI extension config.",
            "LangGraph receives the portable agent contract plus routing, checkpointing, and interrupt intent for production orchestration.",
            "The no-key default proves mock/LangGraph output shape and keeps CrewAI native execution optional until dependency resolution is verified in a compatible environment.",
            "This report is the prototype-to-production bridge: use CrewAI for high-level role/task ideation, then migrate durable workflow concerns to LangGraph.",
            "Hosted APIs, local models, OpenRouter, NVIDIA NIM, and custom OpenAI-compatible gateways remain model routes rather than a custom provider layer.",
        ],
        "validation": {
            "default": "python examples/crewai_prototype_report.py",
            "credentialed": [
                "Install the external CrewAI plugin in Python >=3.10,<3.14.",
                "Install a CrewAI-compatible dependency set and provider credentials for native crew kickoff smoke tests.",
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
                input="Customer says order A123 was double charged.",
                session_id="crewai-prototype-report",
                context={
                    "order_id": "A123",
                    "case_state": "standard",
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
