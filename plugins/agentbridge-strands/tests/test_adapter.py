from __future__ import annotations

import sys
from types import SimpleNamespace

from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge.extensions.strands import StrandsExtension
from agentbridge_strands.adapter import Adapter, CompiledStrandsAgent


class FakeStrandsAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(
            message={"content": [{"text": f"native strands: {prompt}"}]},
            metrics={"requests": 1},
            stop_reason="end_turn",
            structured_output=None,
            interrupts=None,
            checkpoint=None,
        )


def fake_tool(func, **kwargs):
    return SimpleNamespace(func=func, kwargs=kwargs)


def test_adapter_compiles_and_runs_native_agent(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "strands",
        SimpleNamespace(Agent=FakeStrandsAgent, tool=fake_tool),
    )
    adapter = Adapter()

    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return order_id

    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="mock/model",
        tools=[],
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert isinstance(compiled, CompiledStrandsAgent)
    assert compiled.native_agent.kwargs["name"] == "support_agent"
    assert compiled.native_agent.kwargs["system_prompt"] == "Echo the user request."
    assert result.backend == "strands"
    assert result.output == "native strands: hello"
    assert result.usage == {"requests": 1}
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_forwards_strands_extension_surface(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "strands",
        SimpleNamespace(Agent=FakeStrandsAgent, tool=fake_tool),
    )
    adapter = Adapter()
    conversation_manager = object()
    context_manager = object()
    hook = object()
    plugin = object()
    intervention = object()
    session_manager = object()
    memory_manager = object()
    tool_executor = object()
    retry_strategy = object()
    sandbox = object()
    storage = object()
    background_tasks = object()
    agent = StrandsExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Echo the user request.",
            model="mock/model",
        ),
        conversation_manager=conversation_manager,
        context_manager=context_manager,
        hooks=[hook],
        plugins=[plugin],
        interventions=[intervention],
        mcp_clients=["orders_mcp"],
        trace_attributes={"service": "support"},
        guardrails=["refund_policy"],
        session_manager=session_manager,
        memory_manager=memory_manager,
        tool_executor=tool_executor,
        retry_strategy=retry_strategy,
        checkpointing=True,
        sandbox=sandbox,
        storage=storage,
        background_tasks=background_tasks,
        agent_id="agent-1",
        description="Support assistant",
        structured_output_prompt="Return JSON.",
        load_tools_from_directory=True,
        record_direct_tool_call=False,
        deployment_target="agentcore",
        metadata={"owner": "support"},
    )

    compiled = adapter.compile(agent)
    result = adapter.run(
        compiled,
        RunInput(
            input="hello",
            context={"tenant": "acme"},
            metadata={"request_id": "req-1"},
            session_id="session-1",
        ),
    )

    assert compiled.native_agent.kwargs["conversation_manager"] is conversation_manager
    assert compiled.native_agent.kwargs["context_manager"] is context_manager
    assert compiled.native_agent.kwargs["hooks"] == [hook]
    assert compiled.native_agent.kwargs["plugins"] == [plugin]
    assert compiled.native_agent.kwargs["interventions"] == [intervention]
    assert compiled.native_agent.kwargs["trace_attributes"] == {"service": "support"}
    assert compiled.native_agent.kwargs["session_manager"] is session_manager
    assert compiled.native_agent.kwargs["memory_manager"] is memory_manager
    assert compiled.native_agent.kwargs["tool_executor"] is tool_executor
    assert compiled.native_agent.kwargs["retry_strategy"] is retry_strategy
    assert compiled.native_agent.kwargs["checkpointing"] is True
    assert compiled.native_agent.kwargs["sandbox"] is sandbox
    assert compiled.native_agent.kwargs["storage"] is storage
    assert compiled.native_agent.kwargs["background_tasks"] is background_tasks
    assert compiled.native_agent.kwargs["agent_id"] == "agent-1"
    assert compiled.native_agent.kwargs["description"] == "Support assistant"
    assert compiled.native_agent.kwargs["structured_output_prompt"] == "Return JSON."
    assert compiled.native_agent.kwargs["load_tools_from_directory"] is True
    assert compiled.native_agent.kwargs["record_direct_tool_call"] is False
    assert compiled.native_agent.kwargs["state"] == {"metadata": {"owner": "support"}}
    assert result.metadata["invocation_state"] == {
        "tenant": "acme",
        "metadata": {"request_id": "req-1"},
        "session_id": "session-1",
    }
    assert result.metadata["extension_summary"] == {
        "conversation_manager": True,
        "context_manager": True,
        "hooks_count": 1,
        "plugins_count": 1,
        "interventions_count": 1,
        "mcp_clients": ["orders_mcp"],
        "trace_attributes": {"service": "support"},
        "guardrails": ["refund_policy"],
        "deployment_target": "agentcore",
        "session_manager": True,
        "memory_manager": True,
        "applied_native_options": [
            "conversation_manager",
            "context_manager",
            "trace_attributes",
            "hooks",
            "plugins",
            "interventions",
            "session_manager",
            "memory_manager",
            "tool_executor",
            "retry_strategy",
            "checkpointing",
            "sandbox",
            "storage",
            "background_tasks",
            "agent_id",
            "description",
            "structured_output_prompt",
            "load_tools_from_directory",
            "record_direct_tool_call",
        ],
    }


def test_adapter_runs_offline_model_through_native_agent() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="agentbridge/offline",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "strands"
    assert result.output == "offline response: hello"
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_runs_offline_tool_loop_through_native_agent() -> None:
    adapter = Adapter()

    def lookup_order(order_id: str) -> str:
        """Look up an order."""

        return f"found:{order_id}"

    spec = AgentSpec(
        name="support_agent",
        instructions="Use the lookup tool.",
        model="agentbridge/offline",
        tools=[ToolSpec.from_function(lookup_order)],
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="A123"))

    event_types = [event.type for event in result.events]
    assert result.output == "offline tool result: found:A123"
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert event_types[-1] == "complete"


def test_adapter_runs_offline_structured_output_through_native_agent() -> None:
    class Decision(BaseModel):
        eligible: bool = True
        reason: str = "ok"

    adapter = Adapter()
    spec = AgentSpec(
        name="decision_agent",
        instructions="Return a decision.",
        model="agentbridge/offline",
        output_type=Decision,
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="decide"))

    assert result.backend == "strands"
    assert isinstance(result.output, Decision)
    assert result.output.eligible is True
    assert result.output.reason == "ok"
    assert result.events[-1].type == "complete"


def test_adapter_capabilities_mark_structured_output_full() -> None:
    capabilities = Adapter().capabilities()

    assert capabilities.status("structured_output") == "full"
