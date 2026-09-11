"""Named tool registry for safe manifest tool resolution."""

from __future__ import annotations

from collections.abc import Callable

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
