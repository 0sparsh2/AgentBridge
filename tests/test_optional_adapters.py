from __future__ import annotations

import pytest

from agentbridge import AgentSpec, get_adapter
from agentbridge.errors import MissingDependencyError


def test_optional_adapters_raise_actionable_missing_dependency_errors() -> None:
    agent = AgentSpec(name="agent", instructions="Do work.", model="openai/gpt-5")

    for backend in ["pydantic_ai", "langgraph"]:
        adapter = get_adapter(backend)
        try:
            adapter.compile(agent)
        except MissingDependencyError as exc:
            assert backend in str(exc)
            assert "pip install" in str(exc)
        except Exception as exc:  # pragma: no cover - optional package may be installed locally
            pytest.fail(f"{backend} raised an unexpected error: {exc}")
