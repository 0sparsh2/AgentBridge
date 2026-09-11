from __future__ import annotations

from agentbridge import compare_backends


def test_compare_backends_sorts_by_capability_fit() -> None:
    comparisons = compare_backends(
        ["workflow.graph", "tools.sync"],
        backends=["mock", "langgraph"],
    )

    assert comparisons[0].backend == "langgraph"
    assert comparisons[0].fit == "strong"
    assert comparisons[1].backend == "mock"
    assert "workflow.graph" in comparisons[1].missing
