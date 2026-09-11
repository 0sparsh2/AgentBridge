from __future__ import annotations

from agentbridge import AgentManifest, ToolRegistry, validate_manifest


def lookup_order(order_id: str) -> str:
    """Look up an order."""

    return f"found:{order_id}"


def test_validate_manifest_detects_missing_tools() -> None:
    manifest = AgentManifest(
        name="agent",
        instructions="Do work.",
        model="openai/gpt-5",
        tools=[{"name": "lookup_order"}],
    )

    validation = validate_manifest(manifest, tool_registry=ToolRegistry(), backends=["mock"])

    assert not validation.valid
    assert validation.missing_tools == ["lookup_order"]


def test_validate_manifest_reports_backend_fit() -> None:
    registry = ToolRegistry()
    registry.register(lookup_order)
    manifest = AgentManifest(
        name="agent",
        instructions="Do work.",
        model="openai/gpt-5",
        tools=[{"name": "lookup_order"}],
        required_capabilities=["workflow.graph"],
    )

    validation = validate_manifest(manifest, tool_registry=registry, backends=["mock", "langgraph"])

    assert validation.valid
    assert validation.backend_fit["langgraph"] == "medium"
    assert validation.backend_fit["mock"] == "medium"
