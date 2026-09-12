"""AgentBridge adapter plugin for google_adk."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "google_adk"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "partial",
                "tools.sync": "unsupported",
                "state.session": "extension",
                "state.memory": "extension",
                "workflow.delegation": "extension",
                "deployment.serverless": "extension",
                "evals": "extension",
                "streaming.events": "unsupported",
            },
            notes={
                "agent.instructions": "Scaffold compiles AgentSpec shape before native ADK Agent wiring.",
                "tools.sync": "Map ToolSpec callables to ADK tools before marking supported.",
                "state.session": "Planned through GoogleADKExtension session_service and RunInput.session_id.",
                "state.memory": "Planned through GoogleADKExtension memory_service.",
                "deployment.serverless": "Planned as extension-level deployment metadata.",
            },
        )

    def compile(self, spec: AgentSpec) -> AgentSpec:
        """Translate AgentSpec into native framework objects.

        Replace this with framework-specific construction. Returning the spec keeps
        the generated plugin runnable while the adapter is being developed.
        """

        return spec

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled agent and return a normalized result."""

        spec = compiled if isinstance(compiled, AgentSpec) else None
        output = f"[{self.backend_name}] {run_input.input}"
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=[
                AgentEvent(
                    type="message",
                    backend=self.backend_name,
                    data={
                        "content": output,
                        "agent": spec.name if spec else None,
                    },
                ),
                AgentEvent(
                    type="complete",
                    backend=self.backend_name,
                    data={"output": output},
                ),
            ],
            raw=compiled,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        result = self.run(compiled, run_input)
        yield from result.events
