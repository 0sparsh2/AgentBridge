from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec, get_adapter, resume_agent
from agentbridge.errors import MissingDependencyError
from agentbridge.extensions.langgraph import LangGraphConfig, LangGraphExtension
from agentbridge.adapters.langgraph import LangGraphAdapter, LangGraphCompiledAgent


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


def test_langgraph_adapter_accepts_prebuilt_native_graph() -> None:
    class NativeGraph:
        def invoke(self, payload, config=None):
            assert payload["input"] == "A123"
            assert config is None
            return {"output": {"status": "approved"}, "route": "native"}

    agent = LangGraphExtension.with_config(
        AgentSpec(name="native_agent", instructions="Use native graph.", model="openai/gpt-5"),
        native_graph=NativeGraph(),
        native_options={"stream_mode": "updates"},
    )
    adapter = LangGraphAdapter()
    compiled = adapter.compile(agent)
    result = adapter.run(compiled, RunInput(input="A123"))

    assert result.output == {"status": "approved"}
    assert result.metadata["native_graph"] is True
    assert result.metadata["native_options"] == {"stream_mode": "updates"}


def test_langgraph_adapter_passes_runtime_context_to_native_graph() -> None:
    class NativeGraph:
        def invoke(self, payload, config=None, *, context=None):
            assert payload["input"] == "A123"
            assert config is None
            return {"output": {"tenant": context["tenant"]}}

    agent = LangGraphExtension.with_config(
        AgentSpec(name="context_agent", instructions="Use context.", model="openai/gpt-5"),
        native_graph=NativeGraph(),
    )
    result = LangGraphAdapter().run(
        LangGraphAdapter().compile(agent),
        RunInput(input="A123", context={"tenant": "support"}),
    )

    assert result.output == {"tenant": "support"}


def test_langgraph_adapter_invokes_supplied_native_model() -> None:
    class FakeModel:
        def invoke(self, messages):
            assert messages == [{"role": "user", "content": "A123"}]
            return {"content": "Native model approved the refund."}

    agent = LangGraphExtension.with_config(
        AgentSpec(name="model_agent", instructions="Check refunds.", model="vendor/model"),
        model=FakeModel(),
    )
    result = LangGraphAdapter().run(
        LangGraphAdapter().compile(agent),
        RunInput(input="A123"),
    )

    assert result.output["message"] == "Native model approved the refund."
    assert result.output["model_type"] == "FakeModel"
    assert result.metadata["native_model"] is True


def test_langgraph_extension_preserves_custom_persistence_components() -> None:
    checkpointer = object()
    store = object()
    cache = object()
    config = LangGraphExtension.config(
        checkpointer=checkpointer,
        store=store,
        cache=cache,
    )
    parsed = LangGraphConfig.model_validate(config)

    assert parsed.checkpointer is checkpointer
    assert parsed.store is store
    assert parsed.cache is cache
    assert LangGraphAdapter()._build_checkpointer(parsed) is checkpointer


def test_langgraph_adapter_consumes_native_async_graph_stream() -> None:
    class FakeGraph:
        async def astream(self, payload, config=None, stream_mode=None, version=None):
            assert payload["input"] == "A123"
            assert config is None
            assert stream_mode == ["updates", "messages", "custom"]
            assert version == "v2"
            yield {"agent": {"output": {"status": "approved"}}}

    spec = AgentSpec(name="refund_agent", instructions="Check refunds.", model="openai/gpt-5")
    compiled = LangGraphCompiledAgent(
        spec=spec,
        graph=FakeGraph(),
        config=LangGraphConfig(),
    )

    async def collect():
        return [
            event
            async for event in LangGraphAdapter().astream(
                compiled,
                RunInput(input="A123"),
            )
        ]

    events = asyncio.run(collect())
    assert events[-1].type == "complete"
    assert events[-1].data["output"] == {"status": "approved"}


def test_langgraph_adapter_uses_native_async_invoke():
    class FakeGraph:
        async def ainvoke(self, payload, config=None, *, context=None):
            assert payload["input"] == "A123"
            assert config is None
            assert context == {"tenant": "support"}
            return {"output": {"status": "approved"}, "route": "native"}

    spec = AgentSpec(name="refund_agent", instructions="Check refunds.", model="openai/gpt-5")
    compiled = LangGraphCompiledAgent(spec=spec, graph=FakeGraph(), config=LangGraphConfig())

    async def run():
        return await LangGraphAdapter().arun(
            compiled,
            RunInput(input="A123", context={"tenant": "support"}),
        )

    result = asyncio.run(run())
    assert result.output == {"status": "approved"}
    assert result.metadata["async"] is True
    assert result.metadata["route"] == "native"


def test_langgraph_adapter_consumes_native_sync_stream_options_and_subgraphs() -> None:
    class FakeGraph:
        def stream(self, payload, config=None, **options):
            assert payload["input"] == "hello"
            assert config is None
            assert options["stream_mode"] == ["updates"]
            assert options["subgraphs"] is True
            yield (("child",), {"agent": {"output": "from child"}})

    compiled = LangGraphCompiledAgent(
        spec=AgentSpec(name="stream_agent", instructions="Stream.", model="agentbridge/offline"),
        graph=FakeGraph(),
        config=LangGraphConfig(native_options={"stream_mode": ["updates"], "subgraphs": True}),
    )

    events = list(
        LangGraphAdapter().stream(
            compiled,
            RunInput(input="hello"),
        )
    )

    assert events[-1].type == "complete"
    assert events[-1].data["output"] == "from child"
    assert any(event.data.get("namespace") == ("child",) for event in events)


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
        remote_graph="refund_graph_remote",
        deployment_url="https://example.invalid/langgraph",
        deployment={"target": "langgraph_platform"},
        agentcore_memory_id="memory-123",
        agentcore_store_namespace="refunds",
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
    assert result.metadata["deployment"]["remote_graph"] == "refund_graph_remote"
    assert result.metadata["deployment"]["target"] == "langgraph_platform"
    assert result.metadata["agentcore_memory_id"] == "memory-123"
    assert result.metadata["agentcore_store_namespace"] == "refunds"


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


def test_langgraph_adapter_exposes_state_history_updates_and_replay() -> None:
    class FakeGraph:
        def __init__(self):
            self.calls = []

        def get_state(self, config):
            self.calls.append(("get_state", config))
            return {"values": {"approved": False}, "config": config}

        def get_state_history(self, config):
            self.calls.append(("get_state_history", config))
            return iter([{"values": {"approved": False}}])

        def update_state(self, config, values, **kwargs):
            self.calls.append(("update_state", config, values, kwargs))
            return {"config": config, "values": values, **kwargs}

        def invoke(self, payload, config):
            self.calls.append(("invoke", payload, config))
            return {"output": {"replayed": True}}

    graph = FakeGraph()
    compiled = LangGraphCompiledAgent(
        spec=AgentSpec(name="state_agent", instructions="Inspect state.", model="agentbridge/offline"),
        graph=graph,
        config=LangGraphConfig(enable_checkpointing=True),
    )
    adapter = LangGraphAdapter()
    run_input = RunInput(input="inspect", session_id="state-session")

    state = adapter.get_state(compiled, run_input, checkpoint_id="cp-1")
    history = list(adapter.get_state_history(compiled, run_input))
    updated = adapter.update_state(
        compiled,
        run_input,
        values={"approved": True},
        as_node="review",
    )
    replayed = adapter.replay(compiled, run_input, checkpoint_id="cp-1")

    assert state["values"]["approved"] is False
    assert history == [{"values": {"approved": False}}]
    assert updated["values"] == {"approved": True}
    assert updated["as_node"] == "review"
    assert replayed.output == {"replayed": True}
    assert graph.calls[0][1]["configurable"] == {
        "thread_id": "state-session",
        "checkpoint_id": "cp-1",
    }
    assert graph.calls[-1][2]["configurable"]["checkpoint_id"] == "cp-1"


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
    assert capabilities.status("observability.tracing") == "extension"


def test_langgraph_forwards_observability_runtime_config_when_available() -> None:
    adapter = get_adapter("langgraph")
    agent = LangGraphExtension.with_config(
        AgentSpec(
            name="observed_graph",
            instructions="Run an observed graph.",
            model="agentbridge/offline",
        ),
        callbacks=["callback"],
        metadata={"owner": "support"},
        observability={"tags": ["graph"], "run_name": "observed-graph"},
    )

    try:
        compiled = adapter.compile(agent)
    except MissingDependencyError:
        pytest.skip("langgraph optional dependency is not installed")

    invoke_config = adapter._invoke_config(
        compiled,
        RunInput(input="hello", metadata={"request_id": "req-1"}, session_id="session-1"),
    )

    assert invoke_config == {
        "callbacks": ["callback"],
        "metadata": {
            "owner": "support",
            "request_id": "req-1",
            "session_id": "session-1",
        },
        "tags": ["graph"],
        "run_name": "observed-graph",
    }
