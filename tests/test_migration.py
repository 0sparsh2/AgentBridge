from __future__ import annotations

from types import SimpleNamespace

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

        def get_graph(self):
            return SimpleNamespace(nodes={"agent": object(), "tools": object()})

    report = import_langchain_agent(CompiledStateGraph())

    assert report.convertible
    assert "compiled_graph" in report.native_only
    assert any(finding.category == "workflow.graph" for finding in report.findings)


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
