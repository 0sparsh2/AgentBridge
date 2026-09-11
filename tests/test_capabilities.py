from __future__ import annotations

from agentbridge import BackendCapabilities, inspect_backend, inspect_backends


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
