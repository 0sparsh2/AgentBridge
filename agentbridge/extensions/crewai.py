"""CrewAI-specific extension helpers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension
from agentbridge.types import AgentSpec


CrewAIProcess = Literal["sequential", "hierarchical"]


class CrewAIConfig(BaseModel):
    """AgentBridge config for CrewAI-native role/task/crew behavior."""

    role: str | None = None
    goal: str | None = None
    backstory: str | None = None
    task_description: str = "{input}"
    expected_output: str = "A useful response to the user's request."
    process: CrewAIProcess = "sequential"
    verbose: bool = False
    allow_delegation: bool = False
    memory: bool = False
    human_input: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrewAIExtension(FrameworkExtension):
    """Extension namespace for CrewAI-native behavior."""

    framework = "crewai"

    @staticmethod
    def config(
        *,
        role: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        task_description: str = "{input}",
        expected_output: str = "A useful response to the user's request.",
        process: CrewAIProcess = "sequential",
        verbose: bool = False,
        allow_delegation: bool = False,
        memory: bool = False,
        human_input: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable CrewAI adapter configuration."""

        return CrewAIConfig(
            role=role,
            goal=goal,
            backstory=backstory,
            task_description=task_description,
            expected_output=expected_output,
            process=process,
            verbose=verbose,
            allow_delegation=allow_delegation,
            memory=memory,
            human_input=human_input,
            metadata=metadata or {},
        ).model_dump(exclude_none=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with CrewAI extension config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["crewai"] = CrewAIExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})

    def crews(self) -> dict[str, Any]:
        """Return metadata for native CrewAI crew helpers."""

        raw = self.require_raw()
        return {
            "framework": self.framework,
            "raw_type": type(raw).__name__,
            "message": "Use CrewAIExtension.config(...) for role/task/crew mapping.",
        }

    def tasks(self) -> None:
        """Placeholder for future role/task helpers."""

        raise UnsupportedExtension("CrewAI task extension is not implemented yet.")
