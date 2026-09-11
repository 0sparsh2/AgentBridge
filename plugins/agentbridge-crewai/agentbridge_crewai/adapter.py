"""CrewAI adapter plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentbridge.adapters.base import BackendAdapter
from agentbridge.errors import MissingDependencyError
from agentbridge.extensions.crewai import CrewAIConfig
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CrewAICompiledAgent:
    spec: AgentSpec
    agent: Any
    task: Any
    crew: Any
    config: CrewAIConfig


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
                "workflow.delegation": "extension",
                "state.session": "partial",
                "memory.long_term": "extension",
                "streaming.events": "partial",
                "human_approval": "extension",
                "observability.raw": "full",
                "agui.events": "partial",
            },
            notes={
                "workflow.roles_tasks": "CrewAI maps naturally to role, goal, task, and crew concepts.",
                "workflow.delegation": "Configure through CrewAIExtension.config(allow_delegation=True).",
                "memory.long_term": "Configure through CrewAIExtension.config(memory=True).",
                "tools.sync": "ToolSpec conversion is not yet complete for all CrewAI tool variants.",
                "human_approval": "CrewAI-specific flows should live in an extension namespace.",
            },
        )

    def compile(self, spec: AgentSpec) -> CrewAICompiledAgent:
        try:
            from crewai import Agent, Crew, Task
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "crewai", "external crewai plugin") from exc

        config = CrewAIConfig.model_validate(spec.backend_config.get("crewai", {}))
        agent = Agent(
            role=config.role or spec.name,
            goal=config.goal or spec.instructions,
            backstory=(
                config.backstory
                or spec.metadata.get("backstory")
                or "AgentBridge generated CrewAI agent."
            ),
            llm=spec.model,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
        )
        task = Task(
            description=config.task_description,
            expected_output=config.expected_output,
            agent=agent,
            human_input=config.human_input,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=config.verbose,
            process=_resolve_process(config.process),
            memory=config.memory,
        )
        return CrewAICompiledAgent(spec=spec, agent=agent, task=task, crew=crew, config=config)

    def run(self, compiled: CrewAICompiledAgent, run_input: RunInput) -> RunResult:
        raw = compiled.crew.kickoff(inputs={"input": run_input.input, **run_input.context})
        output = str(raw)
        events = [
            AgentEvent(type="message", backend=self.backend_name, data={"role": "assistant", "content": output}),
            AgentEvent(type="complete", backend=self.backend_name, data={"output": output}),
        ]
        metadata = {
            "role": compiled.config.role or compiled.spec.name,
            "process": compiled.config.process,
            "memory": compiled.config.memory,
            "human_input": compiled.config.human_input,
            "extension_metadata": compiled.config.metadata,
        }
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata=metadata,
            raw=raw,
        )


def _resolve_process(process: str) -> Any:
    try:
        from crewai import Process
    except ImportError:  # pragma: no cover - compile() already guards imports
        return process
    return getattr(Process, process, process)
