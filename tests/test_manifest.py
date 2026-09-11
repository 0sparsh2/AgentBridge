from __future__ import annotations

import pytest

from agentbridge import AgentManifest, ToolRegistry, load_agent_spec, load_manifest


def lookup_order(order_id: str) -> str:
    """Look up an order."""

    return f"found:{order_id}"


def test_load_yaml_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "agent.yaml"
    manifest_path.write_text(
        """
name: refund_agent
instructions: Decide refund eligibility.
model: openai/gpt-5
tools:
  - name: lookup_order
required_capabilities:
  - tools.sync
metadata:
  owner: support
""".strip()
    )

    registry = ToolRegistry()
    registry.register(lookup_order)
    manifest = load_manifest(manifest_path)
    spec = load_agent_spec(manifest_path, tool_registry=registry)

    assert isinstance(manifest, AgentManifest)
    assert manifest.required_capabilities == ["tools.sync"]
    assert spec.name == "refund_agent"
    assert spec.tools[0].name == "lookup_order"
    assert spec.metadata["owner"] == "support"


def test_manifest_rejects_unregistered_static_tools() -> None:
    manifest = AgentManifest(
        name="agent",
        instructions="Do work.",
        model="openai/gpt-5",
        tools=[{"name": "lookup"}],
    )

    with pytest.raises(ValueError, match="ToolRegistry"):
        manifest.to_agent_spec()
