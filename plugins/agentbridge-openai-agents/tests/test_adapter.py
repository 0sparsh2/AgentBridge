from __future__ import annotations

import sys
from types import SimpleNamespace

from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge.extensions.openai_agents import OpenAIAgentsExtension
from agentbridge_openai_agents.adapter import Adapter, CompiledOpenAIAgentsAgent


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResult:
    final_output = "native output"
    usage = {"requests": 1}
    new_items = [SimpleNamespace(name="message", content="native output")]


class FakeRunner:
    last_kwargs = None

    @staticmethod
    def run_sync(agent, input, **kwargs):
        FakeRunner.last_kwargs = kwargs
        return FakeResult()


def fake_function_tool(func, **kwargs):
    return SimpleNamespace(func=func, kwargs=kwargs)


def test_adapter_compiles_and_runs_native_agent(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "agents",
        SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner, function_tool=fake_function_tool),
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
    result = adapter.run(compiled, RunInput(input="hello", session_id="session-1"))

    assert isinstance(compiled, CompiledOpenAIAgentsAgent)
    assert compiled.native_agent.kwargs["name"] == "support_agent"
    assert compiled.native_agent.kwargs["instructions"] == "Echo the user request."
    assert result.backend == "openai_agents"
    assert result.output == "native output"
    assert result.usage == {"requests": 1}
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_forwards_openai_agents_extension_surface(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "agents",
        SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner, function_tool=fake_function_tool),
    )
    adapter = Adapter()
    handoff = object()
    mcp_server = object()
    mcp_config = object()
    prompt = object()
    model_settings = object()
    input_guardrail = object()
    output_guardrail = object()
    agent_hooks = object()
    run_hooks = object()
    run_config = object()
    error_handlers = object()
    session = object()
    context = {"tenant": "acme"}
    agent = OpenAIAgentsExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Echo the user request.",
            model="mock/model",
        ),
        handoff_description="Escalate to billing.",
        handoffs=[handoff],
        mcp_servers=[mcp_server],
        mcp_config=mcp_config,
        prompt=prompt,
        model_settings=model_settings,
        input_guardrails=[input_guardrail],
        output_guardrails=[output_guardrail],
        guardrails=["legacy_guardrail"],
        hooks=agent_hooks,
        tool_use_behavior="stop_on_first_tool",
        reset_tool_choice=False,
        tracing=True,
        approval_policy={"refunds": "required"},
        session_id="fallback-session",
        context=context,
        max_turns=4,
        run_hooks=run_hooks,
        run_config=run_config,
        error_handlers=error_handlers,
        previous_response_id="resp-1",
        auto_previous_response_id=True,
        conversation_id="configured-conversation",
        session=session,
        metadata={"owner": "support"},
    )

    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="hello", session_id="runtime-session"))

    assert compiled.native_agent.kwargs["handoff_description"] == "Escalate to billing."
    assert compiled.native_agent.kwargs["handoffs"] == [handoff]
    assert compiled.native_agent.kwargs["mcp_servers"] == [mcp_server]
    assert compiled.native_agent.kwargs["mcp_config"] is mcp_config
    assert compiled.native_agent.kwargs["prompt"] is prompt
    assert compiled.native_agent.kwargs["model_settings"] is model_settings
    assert compiled.native_agent.kwargs["input_guardrails"] == [input_guardrail]
    assert compiled.native_agent.kwargs["output_guardrails"] == [output_guardrail]
    assert compiled.native_agent.kwargs["hooks"] is agent_hooks
    assert compiled.native_agent.kwargs["tool_use_behavior"] == "stop_on_first_tool"
    assert compiled.native_agent.kwargs["reset_tool_choice"] is False
    assert FakeRunner.last_kwargs == {
        "context": context,
        "max_turns": 4,
        "hooks": run_hooks,
        "error_handlers": error_handlers,
        "previous_response_id": "resp-1",
        "auto_previous_response_id": True,
        "conversation_id": "runtime-session",
        "session": session,
        "run_config": run_config,
    }
    assert result.metadata["runner_kwargs"] == {
        "context": context,
        "max_turns": 4,
        "hooks": "object",
        "error_handlers": "object",
        "previous_response_id": "resp-1",
        "auto_previous_response_id": True,
        "conversation_id": "runtime-session",
        "session": "object",
        "run_config": "object",
    }
    assert result.metadata["extension_summary"] == {
        "handoffs_count": 1,
        "mcp_servers_count": 1,
        "guardrails_count": 1,
        "input_guardrails_count": 1,
        "output_guardrails_count": 1,
        "approval_policy": {"refunds": "required"},
        "tracing": True,
        "metadata": {"owner": "support"},
        "conversation_id": "runtime-session",
        "runner_options": [
            "auto_previous_response_id",
            "context",
            "conversation_id",
            "error_handlers",
            "hooks",
            "max_turns",
            "previous_response_id",
            "run_config",
            "session",
        ],
        "applied_native_agent_options": [
            "handoff_description",
            "handoffs",
            "mcp_servers",
            "mcp_config",
            "prompt",
            "model_settings",
            "input_guardrails",
            "output_guardrails",
            "guardrails",
            "hooks",
            "tool_use_behavior",
            "reset_tool_choice",
        ],
        "applied_native_runner_options": [
            "context",
            "max_turns",
            "run_hooks",
            "run_config",
            "error_handlers",
            "previous_response_id",
            "auto_previous_response_id",
            "conversation_id",
            "session_id",
            "session",
        ],
    }


def test_adapter_runs_offline_model_through_native_runner() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="agentbridge/offline",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "openai_agents"
    assert result.output == "offline response: hello"
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_runs_offline_tool_loop_through_native_runner() -> None:
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


def test_adapter_runs_offline_structured_output_through_native_runner() -> None:
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

    assert result.backend == "openai_agents"
    assert isinstance(result.output, Decision)
    assert result.output.eligible is True
    assert result.output.reason == "ok"
    assert result.events[-1].type == "complete"


def test_adapter_capabilities_mark_structured_output_full() -> None:
    capabilities = Adapter().capabilities()

    assert capabilities.status("structured_output") == "full"
