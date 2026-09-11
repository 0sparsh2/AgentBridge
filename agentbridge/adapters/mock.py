"""Deterministic backend used for tests and examples."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agentbridge.adapters.base import BackendAdapter
from agentbridge.tool_execution import arguments_from_schema
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class MockCompiledAgent:
    spec: AgentSpec


class MockAdapter(BackendAdapter):
    """A no-network backend that echoes inputs and exercises tool/event behavior."""

    backend_name = "mock"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "tools.async": "unsupported",
                "structured_output": "unsupported",
                "workflow.graph": "unsupported",
                "workflow.roles_tasks": "unsupported",
                "state.session": "partial",
                "streaming.events": "full",
                "human_approval": "unsupported",
                "observability.raw": "full",
                "agui.events": "partial",
            },
            notes={
                "agent.model": "Model is recorded but not called by the deterministic mock backend.",
                "state.session": "RunInput.session_id is accepted but no durable state is stored.",
                "agui.events": "Events convert to AG-UI-shaped dictionaries, not a full AG-UI server.",
            },
        )

    def compile(self, spec: AgentSpec) -> MockCompiledAgent:
        return MockCompiledAgent(spec=spec)

    def run(self, compiled: MockCompiledAgent, run_input: RunInput) -> RunResult:
        events = list(self.stream(compiled, run_input))
        final_event = events[-1] if events else None
        output = final_event.data.get("output") if final_event else ""
        return RunResult(output=output, backend=self.backend_name, events=events, raw=compiled)

    def stream(self, compiled: MockCompiledAgent, run_input: RunInput) -> Iterator[AgentEvent]:
        spec = compiled.spec
        yield AgentEvent(
            type="message",
            backend=self.backend_name,
            data={
                "role": "assistant",
                "content": f"{spec.name} received: {run_input.input}",
            },
        )

        tool_outputs: list[dict[str, Any]] = []
        for tool in spec.tools:
            arguments = arguments_from_schema(tool.input_schema, run_input)
            yield AgentEvent(
                type="tool_call",
                backend=self.backend_name,
                data={"name": tool.name, "arguments": arguments},
            )
            try:
                result = tool.call(arguments)
                tool_outputs.append({"name": tool.name, "result": result})
                yield AgentEvent(
                    type="tool_result",
                    backend=self.backend_name,
                    data={"name": tool.name, "result": result},
                )
            except Exception as exc:  # pragma: no cover - defensive normalization
                yield AgentEvent(
                    type="error",
                    backend=self.backend_name,
                    data={"name": tool.name, "error": str(exc)},
                )
                raise

        output = {
            "agent": spec.name,
            "input": run_input.input,
            "tools": tool_outputs,
            "message": "Mock backend completed successfully.",
        }
        yield AgentEvent(type="complete", backend=self.backend_name, data={"output": output})
