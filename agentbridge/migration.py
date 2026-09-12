"""Migration helpers for existing framework-native agent objects.

These helpers are intentionally conservative. They extract an AgentSpec only when
common object attributes are visible and return a report that names what remains
extension-level or native-only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.types import AgentSpec, ToolSpec


class MigrationFinding(BaseModel):
    """One migration observation."""

    category: str
    message: str
    severity: str = "info"


class MigrationReport(BaseModel):
    """Result of inspecting a framework-native object for migration."""

    source_framework: str
    source_type: str
    agent_spec: AgentSpec | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    extension_hints: dict[str, Any] = Field(default_factory=dict)
    native_only: list[str] = Field(default_factory=list)
    findings: list[MigrationFinding] = Field(default_factory=list)

    @property
    def convertible(self) -> bool:
        """Whether the report produced a usable AgentSpec."""

        return self.agent_spec is not None


def import_langchain_agent(agent: Any, *, name: str | None = None) -> MigrationReport:
    """Inspect a LangChain agent/runnable and return a migration report."""

    source_type = type(agent).__name__
    model = _string_attr(agent, "model") or _nested_string_attr(agent, "kwargs", "model")
    instructions = (
        _string_attr(agent, "system_prompt")
        or _nested_string_attr(agent, "kwargs", "system_prompt")
        or _string_attr(agent, "prompt")
        or "Migrated LangChain agent. Review and replace these instructions."
    )
    tools = _tool_specs_from_any(getattr(agent, "tools", None) or _nested_attr(agent, "kwargs", "tools"))
    findings = [
        MigrationFinding(
            category="framework",
            message="LangChain agents are often compiled LangGraph graphs; preserve raw graph behavior when in doubt.",
        )
    ]
    native_only: list[str] = []
    extension_hints: dict[str, Any] = {"langchain": {}}

    for attr, capability in [
        ("middleware", "middleware"),
        ("callbacks", "callbacks"),
        ("memory", "memory"),
        ("retrievers", "retrievers"),
    ]:
        value = getattr(agent, attr, None) or _nested_attr(agent, "kwargs", attr)
        if value:
            extension_hints["langchain"][capability] = _safe_summary(value)

    if hasattr(agent, "get_graph") or "CompiledStateGraph" in source_type:
        native_only.append("compiled_graph")
        findings.append(
            MigrationFinding(
                category="workflow.graph",
                message="Compiled graph topology should be reviewed manually or migrated with LangGraph helpers.",
            )
        )

    agent_spec = AgentSpec(
        name=name or _string_attr(agent, "name") or _nested_string_attr(agent, "kwargs", "name") or "migrated_langchain_agent",
        instructions=instructions,
        model=model or "openai/gpt-5",
        tools=tools,
        backend_config={key: value for key, value in extension_hints.items() if value},
    )

    return MigrationReport(
        source_framework="langchain",
        source_type=source_type,
        agent_spec=agent_spec,
        required_capabilities=_required_capabilities(agent_spec),
        extension_hints=extension_hints,
        native_only=native_only,
        findings=findings,
    )


def import_langgraph_graph(graph: Any, *, name: str | None = None) -> MigrationReport:
    """Inspect a LangGraph graph/compiled graph and return a migration report."""

    source_type = type(graph).__name__
    graph_name = name or _string_attr(graph, "name") or _nested_string_attr(graph, "kwargs", "name")
    nodes = _graph_nodes(graph)
    findings = [
        MigrationFinding(
            category="workflow.graph",
            message="LangGraph migration preserves graph intent as extension hints; node behavior may remain native-only.",
        )
    ]
    native_only = ["graph_topology"] if nodes else ["unknown_graph_topology"]
    extension_hints = {
        "langgraph": {
            "graph_name": graph_name or "migrated_langgraph_graph",
            "nodes": nodes,
        }
    }
    agent_spec = AgentSpec(
        name=graph_name or "migrated_langgraph_graph",
        instructions=(
            _string_attr(graph, "instructions")
            or "Migrated LangGraph graph. Review nodes, state, edges, interrupts, and checkpoints."
        ),
        model=_string_attr(graph, "model") or "openai/gpt-5",
        backend_config=extension_hints,
    )
    return MigrationReport(
        source_framework="langgraph",
        source_type=source_type,
        agent_spec=agent_spec,
        required_capabilities=["agent.instructions", "workflow.graph"],
        extension_hints=extension_hints,
        native_only=native_only,
        findings=findings,
    )


def _tool_specs_from_any(value: Any) -> list[ToolSpec]:
    if not value:
        return []
    tools: list[ToolSpec] = []
    for item in value:
        func = getattr(item, "func", None) or getattr(item, "coroutine", None)
        if callable(func):
            tools.append(
                ToolSpec.from_function(
                    func,
                    name=getattr(item, "name", None),
                    description=getattr(item, "description", None),
                )
            )
        elif callable(item):
            tools.append(ToolSpec.from_function(item))
    return tools


def _required_capabilities(spec: AgentSpec) -> list[str]:
    required = ["agent.instructions"]
    if spec.tools:
        required.append("tools.sync")
    if spec.output_schema:
        required.append("structured_output")
    return required


def _graph_nodes(graph: Any) -> list[str]:
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return sorted(str(key) for key in nodes)
    if isinstance(nodes, list | tuple | set):
        return sorted(str(item) for item in nodes)
    get_graph = getattr(graph, "get_graph", None)
    if callable(get_graph):
        try:
            native_graph = get_graph()
            native_nodes = getattr(native_graph, "nodes", None)
            if isinstance(native_nodes, dict):
                return sorted(str(key) for key in native_nodes)
        except Exception:  # pragma: no cover - defensive introspection
            return []
    return []


def _nested_attr(obj: Any, parent: str, child: str) -> Any:
    parent_value = getattr(obj, parent, None)
    if isinstance(parent_value, dict):
        return parent_value.get(child)
    return getattr(parent_value, child, None)


def _string_attr(obj: Any, attr: str) -> str | None:
    value = getattr(obj, attr, None)
    return value if isinstance(value, str) and value else None


def _nested_string_attr(obj: Any, parent: str, child: str) -> str | None:
    value = _nested_attr(obj, parent, child)
    return value if isinstance(value, str) and value else None


def _safe_summary(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    return type(value).__name__
