"""Backend adapter registry."""

from __future__ import annotations

from agentbridge.adapters import LangGraphAdapter, MockAdapter, PydanticAIAdapter
from agentbridge.adapters.base import BackendAdapter
from agentbridge.errors import AdapterNotFoundError, AdapterRegistrationError
from agentbridge.types import BackendCapabilities

_BUILTIN_ADAPTERS: dict[str, type[BackendAdapter]] = {
    MockAdapter.backend_name: MockAdapter,
    PydanticAIAdapter.backend_name: PydanticAIAdapter,
    LangGraphAdapter.backend_name: LangGraphAdapter,
}
_ADAPTERS: dict[str, type[BackendAdapter]] = dict(_BUILTIN_ADAPTERS)
_ADAPTER_SOURCES: dict[str, str] = {name: "builtin" for name in _BUILTIN_ADAPTERS}


def register_adapter(
    adapter: type[BackendAdapter],
    *,
    source: str = "runtime",
    replace: bool = False,
) -> None:
    """Register a custom backend adapter class."""

    if not adapter.backend_name or adapter.backend_name == "base":
        raise ValueError("adapter.backend_name must be set")
    if not issubclass(adapter, BackendAdapter):
        raise TypeError("adapter must be a BackendAdapter subclass")
    if adapter.backend_name in _ADAPTERS and not replace:
        existing_source = _ADAPTER_SOURCES.get(adapter.backend_name, "unknown")
        raise AdapterRegistrationError(
            f"Adapter '{adapter.backend_name}' is already registered from {existing_source}. "
            "Pass replace=True to override it intentionally."
        )
    _ADAPTERS[adapter.backend_name] = adapter
    _ADAPTER_SOURCES[adapter.backend_name] = source


def get_adapter(backend: str) -> BackendAdapter:
    """Return an adapter instance for a backend name."""

    _ensure_plugins_loaded()
    try:
        adapter_type = _ADAPTERS[backend]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise AdapterNotFoundError(f"Unknown backend '{backend}'. Available: {available}") from exc
    return adapter_type()


def list_adapters() -> list[str]:
    """List registered backend names."""

    _ensure_plugins_loaded()
    return sorted(_ADAPTERS)


def adapter_sources() -> dict[str, str]:
    """Return adapter source metadata."""

    _ensure_plugins_loaded()
    return dict(sorted(_ADAPTER_SOURCES.items()))


def is_builtin_adapter(backend: str) -> bool:
    """Return whether a backend is provided by core AgentBridge."""

    return backend in _BUILTIN_ADAPTERS


def reset_adapters() -> None:
    """Reset adapter registry to built-ins. Intended for tests and dev tooling."""

    _ADAPTERS.clear()
    _ADAPTERS.update(_BUILTIN_ADAPTERS)
    _ADAPTER_SOURCES.clear()
    _ADAPTER_SOURCES.update({name: "builtin" for name in _BUILTIN_ADAPTERS})


def inspect_backend(backend: str) -> BackendCapabilities:
    """Return capability metadata for one backend."""

    return get_adapter(backend).capabilities()


def inspect_backends() -> dict[str, BackendCapabilities]:
    """Return capability metadata for every registered backend."""

    return {backend: inspect_backend(backend) for backend in list_adapters()}


def _ensure_plugins_loaded() -> None:
    from agentbridge.plugins import load_adapter_plugins

    load_adapter_plugins()
