"""AgentBridge adapter plugin for strands."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "strands"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "partial",
                "tools.sync": "unsupported",
                "tools.mcp": "extension",
                "structured_output": "extension",
                "guardrails": "extension",
                "observability.tracing": "extension",
                "deployment.serverless": "extension",
                "streaming.events": "unsupported",
            },
            notes={
                "agent.instructions": "Scaffold compiles AgentSpec shape before native Strands Agent wiring.",
                "tools.sync": "Map ToolSpec callables to Strands tools before marking supported.",
                "tools.mcp": "Planned through StrandsExtension mcp_clients.",
                "structured_output": "Planned through Strands structured output model/schema support.",
                "observability.tracing": "Planned through trace_attributes and native run metadata.",
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
