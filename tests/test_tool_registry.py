from __future__ import annotations

from agentbridge import ToolRegistry, ToolSpec
from agentbridge.tool_registry import coerce_tool_registry, load_tool_registry


def lookup_order(order_id: str) -> str:
    """Look up an order."""

    return f"found:{order_id}"


def test_coerce_tool_registry_from_dict() -> None:
    registry = coerce_tool_registry({"lookup": lookup_order})

    assert registry.list() == ["lookup"]
    assert registry.get("lookup").call({"order_id": "A123"}) == "found:A123"


def test_coerce_tool_registry_from_list() -> None:
    registry = coerce_tool_registry([ToolSpec.from_function(lookup_order)])

    assert registry.list() == ["lookup_order"]


def test_load_tool_registry_from_module_factory(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "my_tools.py"
    module_path.write_text(
        """
from agentbridge import ToolRegistry


def lookup_order(order_id: str) -> str:
    \"\"\"Look up an order.\"\"\"
    return f"found:{order_id}"


def build_registry():
    registry = ToolRegistry()
    registry.register(lookup_order)
    return registry
""".strip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = load_tool_registry("my_tools:build_registry")

    assert isinstance(registry, ToolRegistry)
    assert registry.list() == ["lookup_order"]
