"""LangGraph-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension
from agentbridge.types import AgentSpec


class LangGraphConfig(BaseModel):
    """AgentBridge config for LangGraph-native graph behavior."""

    node_name: str = Field(default="agent", min_length=1)
    graph_name: str | None = None
    include_context_in_output: bool = False
    enable_checkpointing: bool = False
    interrupt_before: list[str] | None = None
    interrupt_after: list[str] | None = None


class LangGraphExtension(FrameworkExtension):
    """Extension namespace for LangGraph-native behavior."""

    framework = "langgraph"

    @staticmethod
    def config(
        *,
        node_name: str = "agent",
        graph_name: str | None = None,
        include_context_in_output: bool = False,
        enable_checkpointing: bool = False,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build serializable LangGraph adapter configuration."""

        return LangGraphConfig(
            node_name=node_name,
            graph_name=graph_name,
            include_context_in_output=include_context_in_output,
            enable_checkpointing=enable_checkpointing,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        ).model_dump(exclude_none=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with LangGraph extension config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["langgraph"] = LangGraphExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})

    def checkpointing(self) -> dict[str, Any]:
        """Return metadata for future checkpoint/resume helpers."""

        raw = self.require_raw()
        return {
            "framework": self.framework,
            "raw_type": type(raw).__name__,
            "message": "Use enable_checkpointing=True in LangGraphExtension.config().",
        }

    def conditional_routing(self) -> None:
        """Placeholder for future graph routing helpers."""

        raise UnsupportedExtension("LangGraph conditional routing extension is not implemented yet.")
