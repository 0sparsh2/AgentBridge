"""Named tool registry for safe manifest tool resolution."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from agentbridge.types import ToolSpec


class ToolRegistry:
    """Maps manifest tool names to explicitly registered ToolSpec objects."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec | Callable[..., object], *, name: str | None = None) -> ToolSpec:
        """Register a ToolSpec or Python callable."""

        if isinstance(tool, ToolSpec):
            tool_spec = tool
            if name and name != tool_spec.name:
                tool_spec = tool_spec.model_copy(update={"name": name})
        else:
            tool_spec = ToolSpec.from_function(tool, name=name)

        self._tools[tool_spec.name] = tool_spec
        return tool_spec

    def get(self, name: str) -> ToolSpec:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools)) or "none"
            raise KeyError(f"Unknown tool '{name}'. Registered tools: {available}") from exc

    def list(self) -> list[str]:
        """List registered tool names."""

        return sorted(self._tools)


default_tool_registry = ToolRegistry()


def register_tool(tool: ToolSpec | Callable[..., object], *, name: str | None = None) -> ToolSpec:
    """Register a tool in the default registry."""

    return default_tool_registry.register(tool, name=name)


def get_tool(name: str) -> ToolSpec:
    """Get a tool from the default registry."""

    return default_tool_registry.get(name)


def list_tools() -> list[str]:
    """List default registered tools."""

    return default_tool_registry.list()


def load_tool_registry(reference: str) -> ToolRegistry:
    """Load a ToolRegistry from an explicit module reference.

    The reference must be `module:attribute`. The attribute may be:

    - a ToolRegistry instance
    - a callable returning a ToolRegistry
    - a callable returning a dict/list/tuple of callables or ToolSpecs
    - a dict/list/tuple of callables or ToolSpecs
    """

    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Tool registry reference must use 'module:attribute' format.")

    module = importlib.import_module(module_name)
    try:
        value = getattr(module, attribute_name)
    except AttributeError as exc:
        raise ValueError(f"Tool registry attribute not found: {reference}") from exc

    loaded = value() if callable(value) and not isinstance(value, ToolSpec) else value
    return coerce_tool_registry(loaded)


def coerce_tool_registry(value: Any) -> ToolRegistry:
    """Coerce an explicit registry export into a ToolRegistry."""

    if isinstance(value, ToolRegistry):
        return value

    registry = ToolRegistry()
    if isinstance(value, dict):
        for name, tool in value.items():
            registry.register(tool, name=str(name))
        return registry

    if isinstance(value, (list, tuple)):
        for tool in value:
            registry.register(tool)
        return registry

    raise TypeError(
        "Tool registry export must be a ToolRegistry, dict, list, tuple, "
        "or callable returning one of those."
    )
