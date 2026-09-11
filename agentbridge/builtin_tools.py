"""Small built-in tools for examples and CLI smoke tests."""

from __future__ import annotations

from agentbridge.tool_registry import ToolRegistry


def echo(text: str) -> str:
    """Echo text back to the caller."""

    return text


def lookup_order(order_id: str) -> str:
    """Return mock refund eligibility for an order."""

    return f"Order '{order_id}' is eligible for refund review."


def build_builtin_tool_registry() -> ToolRegistry:
    """Create a registry with safe deterministic demo tools."""

    registry = ToolRegistry()
    registry.register(echo)
    registry.register(lookup_order)
    return registry
