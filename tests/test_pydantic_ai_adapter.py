from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec, get_adapter
from agentbridge.errors import MissingDependencyError
from agentbridge.extensions.pydantic_ai import PydanticAIExtension


def lookup_order(order_id: str) -> str:
    """Look up an order."""

    return f"found:{order_id}"


class RefundDecision(BaseModel):
    eligible: bool
    reason: str


def test_pydantic_ai_adapter_runs_with_offline_test_model() -> None:
    adapter = get_adapter("pydantic_ai")
    agent = AgentSpec(
        name="typed_agent",
        instructions="Use tools when useful.",
        model="test",
        tools=[ToolSpec.from_function(lookup_order)],
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("pydantic-ai-slim optional dependency is not installed")

    result = adapter.run(compiled, run_input=RunInput(input="A123"))

    assert result.backend == "pydantic_ai"
    assert result.output
    assert "lookup_order" in result.output
    assert "found:" in result.output
    assert "tool_call" in [event.type for event in result.events]
    assert "tool_result" in [event.type for event in result.events]


def test_pydantic_ai_adapter_returns_typed_output_with_offline_test_model() -> None:
    adapter = get_adapter("pydantic_ai")
    agent = AgentSpec(
        name="refund_decision_agent",
        instructions="Return a refund decision.",
        model="test",
        output_type=RefundDecision,
        backend_config={
            "custom_output_args": {
                "eligible": True,
                "reason": "duplicate charge",
            }
        },
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("pydantic-ai-slim optional dependency is not installed")

    result = adapter.run(compiled, run_input=RunInput(input="Customer was double charged."))

    assert isinstance(result.output, RefundDecision)
    assert result.output.eligible is True
    assert result.output.reason == "duplicate charge"
    assert result.events[-1].type == "complete"


def test_pydantic_ai_adapter_uses_extension_config_with_offline_test_model() -> None:
    adapter = get_adapter("pydantic_ai")
    agent = PydanticAIExtension.with_config(
        AgentSpec(
            name="refund_decision_agent",
            instructions="Return a refund decision.",
            model="test",
            output_type=RefundDecision,
        ),
        retries=2,
        tool_timeout=5,
        metadata={"owner": "support"},
        custom_output_args={
            "eligible": False,
            "reason": "outside refund window",
        },
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("pydantic-ai-slim optional dependency is not installed")

    result = adapter.run(compiled, run_input=RunInput(input="Customer requests refund."))

    assert isinstance(result.output, RefundDecision)
    assert result.output.eligible is False
    assert result.output.reason == "outside refund window"
    assert result.metadata["retries"] == 2
    assert result.metadata["tool_timeout"] == 5
    assert result.metadata["extension_metadata"] == {"owner": "support"}


def test_pydantic_ai_capabilities_include_validation_retry_extension() -> None:
    capabilities = get_adapter("pydantic_ai").capabilities()

    assert capabilities.status("structured_output.validation_retries") == "extension"
