"""Base adapter contract."""

from __future__ import annotations

from collections.abc import Iterator
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

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        """Stream normalized events for a compiled agent."""

        result = self.run(compiled, run_input)
        yield from result.events
