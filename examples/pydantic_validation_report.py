"""Generate a no-key Pydantic AI-to-LangGraph typed validation report."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from agentbridge import AgentSpec, ToolSpec, get_adapter, run_agent
from agentbridge.extensions.langgraph import LangGraphExtension
from agentbridge.extensions.pydantic_ai import PydanticAIExtension
from examples.model_routes import MODEL_ROUTES


class RefundDecision(BaseModel):
    """Typed output contract shared across framework runs."""

    eligible: bool
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    next_action: str = Field(min_length=1)


def lookup_policy(order_id: str) -> str:
    """Return policy evidence for a refund decision."""

    return f"Order {order_id} matches duplicate-charge refund policy."


typed_agent = AgentSpec(
    name="typed_refund_decision_agent",
    instructions="Return a validated refund decision with evidence and next action.",
    model="test",
    tools=[ToolSpec.from_function(lookup_policy)],
    output_type=RefundDecision,
    backend_config={
        "custom_output_args": {
            "eligible": True,
            "reason": "duplicate charge confirmed",
            "confidence": 0.94,
            "next_action": "issue_refund",
        }
    },
    metadata={
        "owner": "support-quality",
        "scenario": "pydantic_ai_structured_validation",
    },
)


def build_report() -> dict[str, object]:
    pydantic_source = PydanticAIExtension.with_config(
        typed_agent,
        retries=3,
        tool_timeout=8,
        metadata={
            "validation_policy": "strict_refund_decision_contract",
            "retry_reason": "repair invalid typed model outputs",
        },
        custom_output_args={
            "eligible": True,
            "reason": "duplicate charge confirmed",
            "confidence": 0.94,
            "next_action": "issue_refund",
        },
    )
    langgraph_target = LangGraphExtension.with_config(
        typed_agent,
        graph_name="typed_refund_validation_graph",
        node_name="validate_refund_decision",
        include_context_in_output=True,
        enable_checkpointing=True,
        route_on_context_key="validation_state",
        routes={
            "valid": "validate_refund_decision",
            "needs_review": "manual_quality_review",
        },
        interrupt_before=["manual_quality_review"],
    )
    return {
        "scenario": "typed refund validation migration",
        "source_framework": "pydantic_ai",
        "target_framework": "langgraph",
        "agent_spec": typed_agent.model_dump(mode="json"),
        "framework_extensions": {
            "pydantic_ai_source": pydantic_source.backend_config["pydantic_ai"],
            "langgraph_target": langgraph_target.backend_config["langgraph"],
        },
        "model_routes": {
            "offline_test_model": "test",
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
            pydantic_source,
            ["mock", "pydantic_ai", "langgraph"],
        ),
        "capability_deltas": {
            "typed_output": "full",
            "pydantic_validation": "full",
            "validation_retries": "extension",
            "tool_timeout": "extension",
            "dependency_injection": "unsupported",
            "graph_checkpointing": "partial",
            "model_provider_routing": "partial",
        },
        "migration_notes": [
            "Pydantic AI is the strongest source framework for Python-native typed outputs and validation retry policy.",
            "LangGraph can preserve the same Pydantic output contract while adding routing, checkpointing, and interrupt intent.",
            "App-facing code should assert normalized RunResult shape plus typed output serialization, not native retry internals.",
            "The no-key default uses Pydantic AI TestModel and LangGraph deterministic structured fixtures.",
            "Hosted APIs, local models, OpenRouter, NVIDIA NIM, and custom OpenAI-compatible gateways stay documented as routes rather than becoming a second model abstraction.",
        ],
        "validation": {
            "default": "python examples/pydantic_validation_report.py",
            "credentialed": [
                "Swap model='test' for a provider route such as openai/gpt-5 or openrouter/openai/gpt-4o-mini.",
                "Configure provider-specific API keys or base URLs for hosted/local model routes.",
                "Add native Pydantic AI validators/dependencies once dependency-injection helpers graduate from unsupported to extension/full.",
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
                input="Decide refund eligibility for order A123.",
                session_id="pydantic-validation-report",
                context={
                    "order_id": "A123",
                    "validation_state": "valid",
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
            "usage": _json_safe(result.usage),
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
