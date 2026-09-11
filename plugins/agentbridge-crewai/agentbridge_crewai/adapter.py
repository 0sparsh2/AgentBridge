"""CrewAI adapter plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentbridge.adapters.base import BackendAdapter
from agentbridge.errors import MissingDependencyError
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CrewAICompiledAgent:
    spec: AgentSpec
    agent: Any
    crew: Any


class CrewAIAdapter(BackendAdapter):
    """Adapter for CrewAI role/task/crew execution."""

    backend_name = "crewai"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "full",
                "tools.sync": "partial",
                "tools.async": "unsupported",
                "structured_output": "partial",
                "workflow.graph": "unsupported",
                "workflow.roles_tasks": "full",
                "state.session": "partial",
                "streaming.events": "partial",
                "human_approval": "extension",
                "observability.raw": "full",
                "agui.events": "partial",
            },
            notes={
                "workflow.roles_tasks": "CrewAI maps naturally to role, goal, task, and crew concepts.",
                "tools.sync": "ToolSpec conversion is not yet complete for all CrewAI tool variants.",
                "human_approval": "CrewAI-specific flows should live in an extension namespace.",
            },
        )

    def compile(self, spec: AgentSpec) -> CrewAICompiledAgent:
        try:
            from crewai import Agent, Crew, Task
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "crewai", "external crewai plugin") from exc

        agent = Agent(
            role=spec.name,
            goal=spec.instructions,
            backstory=spec.metadata.get("backstory", "AgentBridge generated CrewAI agent."),
            llm=spec.model,
            verbose=False,
        )
        task = Task(
            description="{input}",
            expected_output="A useful response to the user's request.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        return CrewAICompiledAgent(spec=spec, agent=agent, crew=crew)

    def run(self, compiled: CrewAICompiledAgent, run_input: RunInput) -> RunResult:
        raw = compiled.crew.kickoff(inputs={"input": run_input.input, **run_input.context})
        output = str(raw)
        events = [
            AgentEvent(type="message", backend=self.backend_name, data={"role": "assistant", "content": output}),
            AgentEvent(type="complete", backend=self.backend_name, data={"output": output}),
        ]
        return RunResult(output=output, backend=self.backend_name, events=events, raw=raw)
