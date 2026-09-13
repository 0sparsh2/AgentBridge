from __future__ import annotations

from types import SimpleNamespace
from typing import TypedDict

import pytest

from agentbridge import import_langchain_agent, import_langgraph_graph


def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return order_id


def test_import_langchain_agent_extracts_spec_and_extension_hints() -> None:
    tool = SimpleNamespace(
        name="lookup_order",
        description="Look up an order.",
        func=lookup_order,
    )
    agent = SimpleNamespace(
        name="support_agent",
        kwargs={
            "model": "openai:gpt-5",
            "system_prompt": "Help support users.",
            "tools": [tool],
            "middleware": ["redaction"],
            "callbacks": ["langsmith"],
        },
    )

    report = import_langchain_agent(agent)

    assert report.convertible
    assert report.agent_spec is not None
    assert report.agent_spec.name == "support_agent"
    assert report.agent_spec.instructions == "Help support users."
    assert report.agent_spec.model == "openai:gpt-5"
    assert report.agent_spec.tools[0].name == "lookup_order"
    assert "tools.sync" in report.required_capabilities
    assert report.extension_hints["langchain"]["middleware"] == ["redaction"]


def test_import_langchain_compiled_graph_marks_native_only() -> None:
    class CompiledStateGraph:
        name = "compiled_support_graph"
        kwargs = {"model": "openai:gpt-5", "system_prompt": "Compiled graph."}
        interrupt_before = ["tools"]

        def get_graph(self):
            return SimpleNamespace(
                nodes={"agent": object(), "tools": object()},
                edges=[("agent", "tools"), ("tools", "agent")],
            )

    report = import_langchain_agent(CompiledStateGraph())

    assert report.convertible
    assert "compiled_graph" in report.native_only
    assert "interrupt_policy" in report.native_only
    assert report.extension_hints["langchain"]["graph"] == {
        "nodes": ["agent", "tools"],
        "edges": [
            {"source": "agent", "target": "tools"},
            {"source": "tools", "target": "agent"},
        ],
    }
    assert report.extension_hints["langchain"]["interrupts"] == {
        "interrupt_before": ["tools"]
    }
    assert any(finding.category == "workflow.graph" for finding in report.findings)


def test_import_langchain_runnable_sequence_reports_native_shape() -> None:
    class InputSchema:
        pass

    class OutputSchema:
        pass

    runnable = SimpleNamespace(
        name="support_chain",
        kwargs={
            "model": "openai:gpt-5",
            "system_prompt": "Route support requests.",
            "steps": [
                SimpleNamespace(name="prompt"),
                SimpleNamespace(name="model"),
                SimpleNamespace(name="parser"),
            ],
            "retriever": SimpleNamespace(name="policy_docs"),
            "store": SimpleNamespace(name="vector_store"),
        },
        invoke=lambda value: value,
        stream=lambda value: iter([value]),
        get_input_schema=lambda: InputSchema,
        get_output_schema=lambda: OutputSchema,
    )

    report = import_langchain_agent(runnable)

    assert report.convertible
    assert "lcel_sequence" in report.native_only
    assert report.extension_hints["langchain"]["runnable_methods"] == [
        "invoke",
        "stream",
    ]
    assert report.extension_hints["langchain"]["runnable_steps"] == [
        "prompt",
        "model",
        "parser",
    ]
    assert report.extension_hints["langchain"]["schemas"] == {
        "input_schema": "InputSchema",
        "output_schema": "OutputSchema",
    }
    assert report.extension_hints["langchain"]["retriever"] == "SimpleNamespace"
    assert report.extension_hints["langchain"]["store"] == "SimpleNamespace"
    assert any(finding.category == "workflow.sequence" for finding in report.findings)


def test_import_real_langchain_runnable_sequence_reports_graph_shape() -> None:
    langchain_core = pytest.importorskip("langchain_core.runnables")
    runnable_lambda = langchain_core.RunnableLambda
    chain = runnable_lambda(lambda value: {"text": value}) | runnable_lambda(
        lambda value: value["text"]
    )

    report = import_langchain_agent(chain, name="real_lcel_chain")

    assert report.convertible
    assert report.agent_spec is not None
    assert report.agent_spec.name == "real_lcel_chain"
    assert "lcel_sequence" in report.native_only
    assert "compiled_graph" in report.native_only
    assert report.extension_hints["langchain"]["runnable_steps"] == [
        "RunnableLambda",
        "RunnableLambda",
    ]
    assert report.extension_hints["langchain"]["schemas"] == {
        "input_schema": "RunnableLambdaInput",
        "output_schema": "RunnableLambdaOutput",
        "config_schema": "RunnableSequenceConfig",
    }
    assert report.extension_hints["langchain"]["graph"]["nodes"] == [
        "Lambda",
        "Lambda",
        "LambdaInput",
        "LambdaOutput",
    ]
    assert {
        (edge["source"], edge["target"])
        for edge in report.extension_hints["langchain"]["graph"]["edges"]
    } == {
        ("LambdaInput", "Lambda"),
        ("Lambda", "Lambda"),
        ("Lambda", "LambdaOutput"),
    }


def test_import_langgraph_graph_returns_graph_extension_hints() -> None:
    graph = SimpleNamespace(
        name="refund_graph",
        nodes={"agent": object(), "refund_tool": object()},
    )

    report = import_langgraph_graph(graph)

    assert report.convertible
    assert report.agent_spec is not None
    assert report.agent_spec.name == "refund_graph"
    assert report.required_capabilities == ["agent.instructions", "workflow.graph"]
    assert report.extension_hints["langgraph"]["nodes"] == ["agent", "refund_tool"]
    assert "graph_topology" in report.native_only


def test_import_real_langgraph_compiled_state_graph_reports_edges_and_checkpointing() -> None:
    langgraph_graph = pytest.importorskip("langgraph.graph")
    checkpoint_memory = pytest.importorskip("langgraph.checkpoint.memory")

    class RefundState(TypedDict):
        input: str

    def decide(state: RefundState) -> RefundState:
        return state

    graph = langgraph_graph.StateGraph(RefundState)
    graph.add_node("decide", decide)
    graph.add_edge(langgraph_graph.START, "decide")
    graph.add_edge("decide", langgraph_graph.END)
    compiled = graph.compile(
        checkpointer=checkpoint_memory.InMemorySaver(),
        interrupt_before=["decide"],
    )

    report = import_langgraph_graph(compiled, name="refund_state_graph")

    assert report.convertible
    assert report.agent_spec is not None
    assert report.agent_spec.name == "refund_state_graph"
    assert report.extension_hints["langgraph"]["nodes"] == [
        "__end__",
        "__start__",
        "decide",
    ]
    assert report.extension_hints["langgraph"]["edges"] == [
        {"source": "__start__", "target": "decide"},
        {"source": "decide", "target": "__end__"},
    ]
    assert report.extension_hints["langgraph"]["interrupts"] == {
        "interrupt_before_nodes": ["decide"]
    }
    assert report.extension_hints["langgraph"]["checkpointer"] == "InMemorySaver"
    assert "graph_topology" in report.native_only
    assert "interrupt_policy" in report.native_only
    assert "checkpointing" in report.native_only
    assert any(finding.category == "state.checkpointing" for finding in report.findings)
