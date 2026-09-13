from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec, get_adapter, resume_agent
from agentbridge.errors import MissingDependencyError
from agentbridge.extensions.langgraph import LangGraphExtension


def lookup_order(order_id: str) -> str:
    """Look up an order."""

    return f"found:{order_id}"


def test_langgraph_adapter_executes_tools_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = AgentSpec(
        name="refund_agent",
        instructions="Check refunds.",
        model="openai/gpt-5",
        tools=[ToolSpec.from_function(lookup_order)],
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    result = adapter.run(compiled, run_input=RunInput(input="A123"))

    assert result.backend == "langgraph"
    assert result.output["tools"][0]["name"] == "lookup_order"
    assert result.output["tools"][0]["result"] == "found:A123"
    assert [event.type for event in result.events] == [
        "workflow",
        "message",
        "tool_call",
        "tool_result",
        "complete",
    ]


def test_langgraph_adapter_uses_extension_config_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = LangGraphExtension.with_config(
        AgentSpec(
            name="refund_agent",
            instructions="Check refunds.",
            model="openai/gpt-5",
        ),
        node_name="refund_node",
        graph_name="refund_graph",
        include_context_in_output=True,
        enable_checkpointing=True,
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    result = adapter.run(
        compiled,
        run_input=RunInput(
            input="A123",
            context={"tenant": "support"},
            session_id="session-1",
        ),
    )

    assert result.backend == "langgraph"
    assert result.output["context"] == {"tenant": "support"}
    assert result.metadata["node_name"] == "refund_node"
    assert result.metadata["checkpointing"] is True


def test_langgraph_adapter_routes_with_extension_config_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = LangGraphExtension.with_config(
        AgentSpec(
            name="router_agent",
            instructions="Route support requests.",
            model="openai/gpt-5",
        ),
        node_name="default_node",
        route_on_context_key="intent",
        routes={"refund": "refund_node", "billing": "billing_node"},
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    result = adapter.run(
        compiled,
        run_input=RunInput(input="Customer needs a refund.", context={"intent": "refund"}),
    )
    events = list(
        adapter.stream(
            compiled,
            RunInput(input="Customer needs billing help.", context={"intent": "billing"}),
        )
    )

    assert result.output["route"] == "refund_node"
    assert result.metadata["route"] == "refund_node"
    assert result.metadata["run_diagnostics"]["route"] == "refund_node"
    assert result.metadata["run_diagnostics"]["route_targets_count"] == 3
    assert result.metadata["run_diagnostics"]["event_counts"]["complete"] == 1
    assert events[0].type == "workflow"
    assert events[1].data == {
        "phase": "route",
        "route_key": "intent",
        "route": "billing",
        "node": "billing_node",
    }
    assert events[-1].type == "complete"


def test_langgraph_adapter_reports_interrupt_state_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = LangGraphExtension.with_config(
        AgentSpec(
            name="approval_agent",
            instructions="Pause before work.",
            model="openai/gpt-5",
        ),
        enable_checkpointing=True,
        interrupt_before=["agent"],
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    result = adapter.run(
        compiled,
        run_input=RunInput(input="Needs approval.", session_id="approval-session"),
    )

    workflow_events = [event for event in result.events if event.type == "workflow"]
    assert result.output["interrupted"] is True
    assert result.output["next"] == ["agent"]
    assert result.metadata["interrupted"] is True
    assert result.metadata["next"] == ["agent"]
    assert result.metadata["checkpoint"]["thread_id"] == "approval-session"
    assert result.metadata["run_diagnostics"]["interrupted"] is True
    assert result.metadata["run_diagnostics"]["checkpoint"]["thread_id"] == "approval-session"
    assert any(event.data["phase"] == "interrupted" for event in workflow_events)


def test_langgraph_adapter_resumes_checkpointed_interrupt_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = LangGraphExtension.with_config(
        AgentSpec(
            name="approval_agent",
            instructions="Pause before work.",
            model="openai/gpt-5",
        ),
        enable_checkpointing=True,
        interrupt_before=["agent"],
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    interrupted = adapter.run(
        compiled,
        run_input=RunInput(input="Needs approval.", session_id="approval-session"),
    )
    resumed = adapter.resume(
        compiled,
        run_input=RunInput(input="Approved.", session_id="approval-session"),
    )
    adapter.run(
        compiled,
        run_input=RunInput(input="Needs helper approval.", session_id="approval-session-2"),
    )
    helper_resumed = resume_agent(
        compiled,
        backend="langgraph",
        input="Approved again.",
        session_id="approval-session-2",
    )

    assert interrupted.metadata["interrupted"] is True
    assert resumed.metadata["resumed"] is True
    assert "interrupted" not in resumed.metadata
    assert resumed.output["route"] == "agent"
    assert resumed.output["input"] == "Needs approval."
    assert resumed.events[0].data["phase"] == "resumed"
    assert helper_resumed.metadata["resumed"] is True
    assert helper_resumed.output["input"] == "Needs helper approval."


def test_langgraph_adapter_returns_typed_structured_output_when_available() -> None:
    class Decision(BaseModel):
        eligible: bool
        reason: str

    adapter = get_adapter("langgraph")
    agent = AgentSpec(
        name="decision_agent",
        instructions="Return a decision.",
        model="openai/gpt-5",
        output_type=Decision,
        backend_config={"custom_output_args": {"eligible": True, "reason": "ok"}},
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    result = adapter.run(compiled, RunInput(input="decide"))

    assert isinstance(result.output, Decision)
    assert result.output.eligible is True
    assert result.output.reason == "ok"
    assert result.metadata["run_diagnostics"]["structured_output"] is True
    assert result.events[-1].data["output"] == Decision(eligible=True, reason="ok")


def test_langgraph_adapter_returns_schema_structured_output_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = AgentSpec(
        name="schema_agent",
        instructions="Return a schema-shaped object.",
        model="openai/gpt-5",
        output_schema={
            "type": "object",
            "properties": {
                "eligible": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["eligible", "reason"],
        },
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    result = adapter.run(compiled, RunInput(input="decide"))

    assert result.output == {"eligible": True, "reason": "ok"}
    assert result.metadata["run_diagnostics"]["structured_output"] is True


def test_langgraph_capabilities_include_checkpointing_extension() -> None:
    capabilities = get_adapter("langgraph").capabilities()

    assert capabilities.status("state.checkpointing") == "extension"
    assert capabilities.status("workflow.routing") == "extension"
    assert capabilities.status("human_approval") == "extension"
    assert capabilities.status("structured_output") == "full"
