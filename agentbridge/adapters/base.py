"""Base adapter contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


class BackendAdapter:
    """Base class for framework adapters."""

    backend_name = "base"

    def capabilities(self) -> BackendCapabilities:
        """Return feature support metadata for this backend."""

        return BackendCapabilities(backend=self.backend_name)

    def compile(self, spec: AgentSpec) -> Any:
        """Compile an AgentSpec into backend-native objects."""

        raise NotImplementedError

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run a compiled agent and return a normalized result."""

        raise NotImplementedError

    async def arun(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run asynchronously; adapters may override with native async execution."""

        return await asyncio.to_thread(self.run, compiled, run_input)

    def resume(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Resume a previously interrupted compiled agent."""

        raise NotImplementedError(f"{self.backend_name} does not implement resume().")

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        """Stream normalized events for a compiled agent."""

        result = self.run(compiled, run_input)
        yield from result.events

    async def astream(self, compiled: Any, run_input: RunInput) -> AsyncIterator[AgentEvent]:
        """Stream normalized events asynchronously without blocking the caller."""

        events = await asyncio.to_thread(lambda: list(self.stream(compiled, run_input)))
        for event in events:
            yield event
