from __future__ import annotations

from agentbridge import BackendCapabilities, capability_matrix, inspect_backend, inspect_backends


def test_backend_capabilities_report_feature_support() -> None:
    capabilities = inspect_backend("langgraph")

    assert isinstance(capabilities, BackendCapabilities)
    assert capabilities.backend == "langgraph"
    assert capabilities.status("workflow.graph") == "full"
    assert capabilities.supports("workflow.graph")
    assert not capabilities.supports("unknown.feature")


def test_all_builtin_backends_report_capabilities() -> None:
    backend_capabilities = inspect_backends()

    assert {"mock", "pydantic_ai", "langgraph"}.issubset(backend_capabilities)
    assert "crewai" not in backend_capabilities
    for backend, capabilities in backend_capabilities.items():
        assert capabilities.backend == backend
        assert "agent.instructions" in capabilities.features
        assert "observability.raw" in capabilities.features


def test_capability_matrix_reports_canonical_features() -> None:
    matrix = capability_matrix(backends=["mock", "langgraph"])

    assert matrix.backends == ["mock", "langgraph"]
    rows = {row.feature.key: row for row in matrix.rows}
    assert rows["agent.instructions"].support == {"mock": "full", "langgraph": "partial"}
    assert rows["workflow.graph"].support["mock"] == "unsupported"
    assert rows["workflow.graph"].support["langgraph"] == "full"
    assert "workflow.graph" in matrix.as_markdown()
    assert "tools.mcp" in rows
    assert "guardrails" in rows
    assert "workflow.handoffs" in rows
    assert "deployment.serverless" in rows
    assert "observability.tracing" in rows
    assert "state.memory" in rows
    assert "evals" in rows
