"""LangGraph-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class LangGraphConfig(BaseModel):
    """AgentBridge config for LangGraph-native graph behavior."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_name: str = Field(default="agent", min_length=1)
    graph_name: str | None = None
    remote_graph: str | None = None
    deployment_url: str | None = None
    deployment: dict[str, Any] = Field(default_factory=dict)
    model: Any | None = None
    native_graph: Any | None = None
    native_options: dict[str, Any] = Field(default_factory=dict)
    stream_options: dict[str, Any] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    checkpointer: Any | None = None
    store: Any | None = None
    cache: Any | None = None
    agentcore_memory_id: str | None = None
    agentcore_store_namespace: str | None = None
    include_context_in_output: bool = False
    enable_checkpointing: bool = False
    route_on_context_key: str | None = None
    routes: dict[str, str] = Field(default_factory=dict)
    interrupt_before: list[str] | None = None
    interrupt_after: list[str] | None = None
    callbacks: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)


class LangGraphExtension(FrameworkExtension):
    """Extension namespace for LangGraph-native behavior."""

    framework = "langgraph"

    @staticmethod
    def config(
        *,
        node_name: str = "agent",
        graph_name: str | None = None,
        remote_graph: str | None = None,
        deployment_url: str | None = None,
        deployment: dict[str, Any] | None = None,
        model: Any | None = None,
        native_graph: Any | None = None,
        native_options: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
        checkpointer: Any | None = None,
        store: Any | None = None,
        cache: Any | None = None,
        agentcore_memory_id: str | None = None,
        agentcore_store_namespace: str | None = None,
        include_context_in_output: bool = False,
        enable_checkpointing: bool = False,
        route_on_context_key: str | None = None,
        routes: dict[str, str] | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callbacks: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        observability: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable LangGraph adapter configuration."""

        return LangGraphConfig(
            node_name=node_name,
            graph_name=graph_name,
            remote_graph=remote_graph,
            deployment_url=deployment_url,
            deployment=deployment or {},
            model=model,
            native_graph=native_graph,
            native_options=native_options or {},
            stream_options=stream_options or {},
            runtime_config=runtime_config or {},
            checkpointer=checkpointer,
            store=store,
            cache=cache,
            agentcore_memory_id=agentcore_memory_id,
            agentcore_store_namespace=agentcore_store_namespace,
            include_context_in_output=include_context_in_output,
            enable_checkpointing=enable_checkpointing,
            route_on_context_key=route_on_context_key,
            routes=routes or {},
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callbacks=callbacks or [],
            metadata=metadata or {},
            observability=observability or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

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

    @staticmethod
    def conditional_routing(
        *,
        context_key: str,
        routes: dict[str, str],
        default_node: str = "agent",
    ) -> dict[str, Any]:
        """Build a validated conditional-routing configuration.

        Route selection reads ``context_key`` from ``RunInput.context``. Values
        not present in ``routes`` use ``default_node`` through the adapter's
        ``__default__`` branch.
        """

        if not context_key.strip():
            raise ValueError("LangGraph routing context_key must not be empty.")
        if not routes:
            raise ValueError("LangGraph routing requires at least one route.")
        if any(not key.strip() or not node.strip() for key, node in routes.items()):
            raise ValueError("LangGraph route keys and node names must not be empty.")
        if not default_node.strip():
            raise ValueError("LangGraph routing default_node must not be empty.")
        return LangGraphExtension.config(
            node_name=default_node,
            route_on_context_key=context_key,
            routes=dict(routes),
        )
