"""Convenience run helpers."""

from __future__ import annotations

from collections.abc import Iterator

from agentbridge.registry import get_adapter
from agentbridge.types import AgentEvent, AgentSpec, RunInput, RunResult


def _coerce_run_input(input: str | RunInput, **kwargs: object) -> RunInput:
    if isinstance(input, RunInput):
        return input
    return RunInput(input=input, **kwargs)


def _resolve_backend(*, backend: str | None, framework: str | None) -> str:
    if backend and framework and backend != framework:
        raise ValueError("Pass either backend or framework, not conflicting values for both.")
    resolved = framework or backend
    if not resolved:
        raise ValueError("A framework/backend name is required.")
    return resolved


def run_agent(
    agent: AgentSpec,
    *,
    backend: str | None = None,
    framework: str | None = None,
    input: str | RunInput,
    **run_input_kwargs: object,
) -> RunResult:
    """Compile and run an agent against a framework adapter."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    compiled = adapter.compile(agent)
    return adapter.run(compiled, _coerce_run_input(input, **run_input_kwargs))


def stream_agent(
    agent: AgentSpec,
    *,
    backend: str | None = None,
    framework: str | None = None,
    input: str | RunInput,
    **run_input_kwargs: object,
) -> Iterator[AgentEvent]:
    """Compile and stream an agent against a framework adapter."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    compiled = adapter.compile(agent)
    yield from adapter.stream(compiled, _coerce_run_input(input, **run_input_kwargs))
