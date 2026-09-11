"""Adapter plugin discovery and loading."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from agentbridge.adapters.base import BackendAdapter
from agentbridge.registry import register_adapter

ENTRY_POINT_GROUP = "agentbridge.adapters"
PLUGIN_ENV_VAR = "AGENTBRIDGE_ADAPTER_PLUGINS"
PLUGIN_OVERRIDE_ENV_VAR = "AGENTBRIDGE_ADAPTER_PLUGIN_OVERRIDES"


@dataclass(frozen=True)
class PluginLoadResult:
    """Result of one plugin load attempt."""

    source: str
    name: str
    loaded: bool
    backend: str | None = None
    error: str | None = None
    replaced: bool = False


_LOADED = False
_RESULTS: list[PluginLoadResult] = []


def load_adapter_plugins(*, force: bool = False) -> list[PluginLoadResult]:
    """Load adapter plugins from entry points and environment modules."""

    global _LOADED, _RESULTS
    if _LOADED and not force:
        return list(_RESULTS)

    results: list[PluginLoadResult] = []
    results.extend(_load_entry_point_plugins())
    results.extend(_load_env_plugins())
    _RESULTS = results
    _LOADED = True
    return list(_RESULTS)


def plugin_status() -> list[PluginLoadResult]:
    """Return plugin load results, loading plugins once if needed."""

    return load_adapter_plugins()


def reset_plugin_loader() -> None:
    """Reset plugin loader state for tests."""

    from agentbridge.registry import reset_adapters

    global _LOADED, _RESULTS
    _LOADED = False
    _RESULTS = []
    reset_adapters()


def _load_entry_point_plugins() -> list[PluginLoadResult]:
    try:
        discovered = entry_points()
        if hasattr(discovered, "select"):
            selected = discovered.select(group=ENTRY_POINT_GROUP)
        else:  # pragma: no cover - old importlib.metadata compatibility
            selected = discovered.get(ENTRY_POINT_GROUP, [])
    except Exception as exc:  # pragma: no cover - defensive discovery guard
        return [
            PluginLoadResult(
                source="entry_point",
                name=ENTRY_POINT_GROUP,
                loaded=False,
                error=str(exc),
            )
        ]

    return [_load_entry_point(entry_point) for entry_point in selected]


def _load_entry_point(entry_point: EntryPoint) -> PluginLoadResult:
    try:
        adapter = entry_point.load()
        backend, replaced = _register_plugin_adapter(adapter, name=f"entry_point:{entry_point.name}")
        return PluginLoadResult(
            source="entry_point",
            name=entry_point.name,
            loaded=True,
            backend=backend,
            replaced=replaced,
        )
    except Exception as exc:
        return PluginLoadResult(
            source="entry_point",
            name=entry_point.name,
            loaded=False,
            error=str(exc),
        )


def _load_env_plugins() -> list[PluginLoadResult]:
    modules = [item.strip() for item in os.environ.get(PLUGIN_ENV_VAR, "").split(",") if item.strip()]
    return [_load_env_module(module) for module in modules]


def _load_env_module(module_name: str) -> PluginLoadResult:
    try:
        module = importlib.import_module(module_name)
        adapter = _adapter_from_module(module)
        backend, replaced = _register_plugin_adapter(adapter, name=f"env:{module_name}")
        return PluginLoadResult(
            source="env",
            name=module_name,
            loaded=True,
            backend=backend,
            replaced=replaced,
        )
    except Exception as exc:
        return PluginLoadResult(source="env", name=module_name, loaded=False, error=str(exc))


def _adapter_from_module(module: Any) -> Any:
    if hasattr(module, "get_adapter"):
        return module.get_adapter()
    if hasattr(module, "Adapter"):
        return module.Adapter
    raise ValueError("Plugin module must expose get_adapter() or Adapter")


def _register_plugin_adapter(adapter: Any, *, name: str) -> tuple[str, bool]:
    if isinstance(adapter, BackendAdapter):
        adapter_type = type(adapter)
    else:
        adapter_type = adapter
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, BackendAdapter):
        raise TypeError("Plugin must return a BackendAdapter subclass or instance")
    replace = adapter_type.backend_name in _plugin_overrides()
    register_adapter(adapter_type, source=name, replace=replace)
    return adapter_type.backend_name, replace


def _plugin_overrides() -> set[str]:
    return {
        item.strip()
        for item in os.environ.get(PLUGIN_OVERRIDE_ENV_VAR, "").split(",")
        if item.strip()
    }
