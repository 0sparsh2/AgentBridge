from __future__ import annotations

import pytest

from agentbridge import adapter_sources, get_adapter, list_adapters, register_adapter, reset_adapters
from agentbridge.adapters import BackendAdapter
from agentbridge.errors import AdapterNotFoundError, AdapterRegistrationError


def test_builtin_adapters_are_registered() -> None:
    assert {"mock", "pydantic_ai", "langgraph"}.issubset(set(list_adapters()))
    assert "crewai" not in list_adapters()


def test_unknown_backend_raises_clear_error() -> None:
    with pytest.raises(AdapterNotFoundError, match="Unknown backend"):
        get_adapter("missing")


def test_custom_adapter_registration() -> None:
    class ExampleAdapter(BackendAdapter):
        backend_name = "example"

    register_adapter(ExampleAdapter)

    assert get_adapter("example").backend_name == "example"
    assert adapter_sources()["example"] == "runtime"


def test_adapter_registration_requires_explicit_replace() -> None:
    class ExampleReplacementAdapter(BackendAdapter):
        backend_name = "example"

    with pytest.raises(AdapterRegistrationError, match="already registered"):
        register_adapter(ExampleReplacementAdapter)

    register_adapter(ExampleReplacementAdapter, source="test", replace=True)

    assert get_adapter("example").__class__ is ExampleReplacementAdapter
    assert adapter_sources()["example"] == "test"
    reset_adapters()
