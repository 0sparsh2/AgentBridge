"""Convenience run helpers."""

from __future__ import annotations

from collections.abc import Iterator

from agentbridge.registry import get_adapter
from agentbridge.types import AgentEvent, AgentSpec, RunInput, RunResult


def _coerce_run_input(input: str | RunInput, **kwargs: object) -> RunInput:
    if isinstance(input, RunInput):
        return input
    return RunInput(input=input, **kwargs)


def run_agent(
    agent: AgentSpec,
    *,
    backend: str,
    input: str | RunInput,
    **run_input_kwargs: object,
) -> RunResult:
    """Compile and run an agent against a backend."""

    adapter = get_adapter(backend)
    compiled = adapter.compile(agent)
    return adapter.run(compiled, _coerce_run_input(input, **run_input_kwargs))


def stream_agent(
    agent: AgentSpec,
    *,
    backend: str,
    input: str | RunInput,
    **run_input_kwargs: object,
) -> Iterator[AgentEvent]:
    """Compile and stream an agent against a backend."""

    adapter = get_adapter(backend)
    compiled = adapter.compile(agent)
    yield from adapter.stream(compiled, _coerce_run_input(input, **run_input_kwargs))
