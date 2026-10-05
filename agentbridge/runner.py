"""Convenience run helpers."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

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


async def arun_agent(
    agent: AgentSpec,
    *,
    backend: str | None = None,
    framework: str | None = None,
    input: str | RunInput,
    **run_input_kwargs: object,
) -> RunResult:
    """Compile and run an agent asynchronously against a framework adapter."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    compiled = adapter.compile(agent)
    return await adapter.arun(compiled, _coerce_run_input(input, **run_input_kwargs))


def batch_agent(
    agent: AgentSpec,
    *,
    backend: str | None = None,
    framework: str | None = None,
    inputs: Iterable[str | RunInput],
    **run_input_kwargs: object,
) -> list[RunResult]:
    """Run multiple inputs through one compiled backend agent."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    compiled = adapter.compile(agent)
    run_inputs = (_coerce_run_input(item, **run_input_kwargs) for item in inputs)
    return adapter.batch(compiled, run_inputs)


async def abatch_agent(
    agent: AgentSpec,
    *,
    backend: str | None = None,
    framework: str | None = None,
    inputs: Iterable[str | RunInput] | AsyncIterable[str | RunInput],
    **run_input_kwargs: object,
) -> list[RunResult]:
    """Run multiple inputs asynchronously through one compiled backend agent."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    compiled = adapter.compile(agent)
    if hasattr(inputs, "__aiter__"):
        run_inputs = [_coerce_run_input(item, **run_input_kwargs) async for item in inputs]  # type: ignore[union-attr]
    else:
        run_inputs = [_coerce_run_input(item, **run_input_kwargs) for item in inputs]  # type: ignore[arg-type]
    return await adapter.abatch(compiled, run_inputs)


def resume_agent(
    compiled: object,
    *,
    backend: str | None = None,
    framework: str | None = None,
    input: str | RunInput = "resume",
    **run_input_kwargs: object,
) -> RunResult:
    """Resume a previously interrupted compiled agent.

    Resume is intentionally based on a compiled backend object because some
    frameworks keep checkpoint state on the compiled runtime.
    """

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    return adapter.resume(compiled, _coerce_run_input(input, **run_input_kwargs))


async def aresume_agent(
    compiled: object,
    *,
    backend: str | None = None,
    framework: str | None = None,
    input: str | RunInput = "resume",
    **run_input_kwargs: object,
) -> RunResult:
    """Resume a previously interrupted compiled agent asynchronously."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    return await adapter.aresume(compiled, _coerce_run_input(input, **run_input_kwargs))


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


async def astream_agent(
    agent: AgentSpec,
    *,
    backend: str | None = None,
    framework: str | None = None,
    input: str | RunInput,
    **run_input_kwargs: object,
) -> AsyncIterator[AgentEvent]:
    """Compile and asynchronously stream normalized adapter events."""

    adapter = get_adapter(_resolve_backend(backend=backend, framework=framework))
    compiled = adapter.compile(agent)
    async for event in adapter.astream(compiled, _coerce_run_input(input, **run_input_kwargs)):
        yield event
