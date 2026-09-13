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
        ("retriever", "retriever"),
        ("checkpointer", "checkpointer"),
        ("store", "store"),
    ]:
        value = getattr(agent, attr, None) or _nested_attr(agent, "kwargs", attr)
        if value:
            extension_hints["langchain"][capability] = _safe_summary(value)

    runnable_methods = _runnable_methods(agent)
    if runnable_methods:
        extension_hints["langchain"]["runnable_methods"] = runnable_methods
        findings.append(
            MigrationFinding(
                category="runtime.runnable",
                message="Runnable execution methods were detected; preserve native runnable behavior until covered by adapter tests.",
            )
        )

    runnable_steps = _runnable_steps(agent)
    if runnable_steps:
        extension_hints["langchain"]["runnable_steps"] = runnable_steps
        native_only.append("lcel_sequence")
        findings.append(
            MigrationFinding(
                category="workflow.sequence",
                message="LCEL sequence steps were detected and should be reviewed before flattening into AgentSpec.",
            )
        )

    schemas = _schema_hints(agent)
    if schemas:
        extension_hints["langchain"]["schemas"] = schemas

    graph_summary = _graph_summary(agent)
    if graph_summary:
        extension_hints["langchain"]["graph"] = graph_summary
        native_only.append("compiled_graph")
        findings.append(
            MigrationFinding(
                category="workflow.graph",
                message="Compiled graph topology should be reviewed manually or migrated with LangGraph helpers.",
            )
        )

    interrupt_summary = _interrupt_hints(agent)
    if interrupt_summary:
        extension_hints["langchain"]["interrupts"] = interrupt_summary
        native_only.append("interrupt_policy")

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
    graph_summary = _graph_summary(graph)
    interrupt_summary = _interrupt_hints(graph)
    checkpointer = getattr(graph, "checkpointer", None) or _nested_attr(graph, "kwargs", "checkpointer")
    findings = [
        MigrationFinding(
            category="workflow.graph",
            message="LangGraph migration preserves graph intent as extension hints; node behavior may remain native-only.",
        )
    ]
    native_only = ["graph_topology"] if nodes else ["unknown_graph_topology"]
    if interrupt_summary:
        native_only.append("interrupt_policy")
        findings.append(
            MigrationFinding(
                category="workflow.interrupts",
                message="LangGraph interrupt policy was detected and should remain native until explicitly remapped.",
            )
        )
    if checkpointer:
        native_only.append("checkpointing")
        findings.append(
            MigrationFinding(
                category="state.checkpointing",
                message="LangGraph checkpointing was detected; preserve checkpointer behavior for resume semantics.",
            )
        )
    extension_hints = {
        "langgraph": {
            "graph_name": graph_name or "migrated_langgraph_graph",
            "nodes": nodes,
        }
    }
    if graph_summary.get("edges"):
        extension_hints["langgraph"]["edges"] = graph_summary["edges"]
    if interrupt_summary:
        extension_hints["langgraph"]["interrupts"] = interrupt_summary
    if checkpointer:
        extension_hints["langgraph"]["checkpointer"] = _safe_summary(checkpointer)
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
    get_graph = getattr(graph, "get_graph", None)
    if callable(get_graph):
        try:
            native_graph = get_graph()
            native_nodes = getattr(native_graph, "nodes", None)
            if isinstance(native_nodes, dict):
                return sorted(
                    str(getattr(node, "name", key))
                    for key, node in native_nodes.items()
                )
        except Exception:  # pragma: no cover - defensive introspection
            return []
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return sorted(str(getattr(node, "name", key)) for key, node in nodes.items())
    if isinstance(nodes, list | tuple | set):
        return sorted(str(item) for item in nodes)
    return []


def _graph_summary(agent: Any) -> dict[str, Any]:
    if not (hasattr(agent, "get_graph") or "CompiledStateGraph" in type(agent).__name__):
        return {}
    summary: dict[str, Any] = {}
    native_graph = _call_no_arg(agent, "get_graph")
    nodes = _graph_nodes(agent)
    if nodes:
        summary["nodes"] = nodes
    edges = _graph_edges(native_graph, _graph_node_name_map(native_graph))
    if edges:
        summary["edges"] = edges
    if not summary and native_graph is not None:
        summary["type"] = type(native_graph).__name__
    return summary or {"detected": True}


def _graph_edges(graph: Any, node_names: dict[str, str] | None = None) -> list[dict[str, str]]:
    if graph is None:
        return []
    node_names = node_names or {}
    edges = getattr(graph, "edges", None)
    if isinstance(edges, dict):
        return [
            {
                "source": node_names.get(str(source), str(source)),
                "target": node_names.get(str(target), str(target)),
            }
            for source, targets in edges.items()
            for target in _iter_targets(targets)
        ]
    if isinstance(edges, list | tuple | set):
        normalized: list[dict[str, str]] = []
        for edge in edges:
            source = getattr(edge, "source", None) or getattr(edge, "from_node", None)
            target = getattr(edge, "target", None) or getattr(edge, "to_node", None)
            if source is None and isinstance(edge, list | tuple) and len(edge) >= 2:
                source, target = edge[0], edge[1]
            if source is not None and target is not None:
                normalized.append(
                    {
                        "source": node_names.get(str(source), str(source)),
                        "target": node_names.get(str(target), str(target)),
                    }
                )
        return normalized
    return []


def _graph_node_name_map(graph: Any) -> dict[str, str]:
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return {}
    return {
        str(node_id): str(getattr(node, "name", node_id))
        for node_id, node in nodes.items()
    }


def _iter_targets(value: Any) -> list[Any]:
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _runnable_methods(agent: Any) -> list[str]:
    return [
        method
        for method in (
            "invoke",
            "ainvoke",
            "batch",
            "abatch",
            "stream",
            "astream",
            "stream_events",
            "astream_events",
        )
        if callable(getattr(agent, method, None))
    ]


def _runnable_steps(agent: Any) -> list[str]:
    steps = getattr(agent, "steps", None) or _nested_attr(agent, "kwargs", "steps")
    if isinstance(steps, list | tuple):
        return [_step_name(step) for step in steps]

    ordered_steps = [
        getattr(agent, attr, None) or _nested_attr(agent, "kwargs", attr)
        for attr in ("first", "middle", "last")
    ]
    flattened: list[str] = []
    for step in ordered_steps:
        if not step:
            continue
        if isinstance(step, list | tuple):
            flattened.extend(_step_name(item) for item in step)
        else:
            flattened.append(_step_name(step))
    return flattened


def _step_name(step: Any) -> str:
    return _string_attr(step, "name") or type(step).__name__


def _schema_hints(agent: Any) -> dict[str, str]:
    hints: dict[str, str] = {}
    for attr in ("input_schema", "output_schema", "config_schema"):
        value = getattr(agent, attr, None) or _nested_attr(agent, "kwargs", attr)
        if value and not callable(value):
            hints[attr] = _safe_type_name(value)
    for method_name, label in [
        ("get_input_schema", "input_schema"),
        ("get_output_schema", "output_schema"),
        ("config_schema", "config_schema"),
    ]:
        if label in hints:
            continue
        value = _call_no_arg(agent, method_name)
        if value:
            hints[label] = _safe_type_name(value)
    return hints


def _interrupt_hints(agent: Any) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for attr in (
        "interrupt_before",
        "interrupt_after",
        "interrupt_before_nodes",
        "interrupt_after_nodes",
    ):
        value = getattr(agent, attr, None) or _nested_attr(agent, "kwargs", attr)
        if value:
            hints[attr] = _safe_summary(value)
    return hints


def _call_no_arg(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:  # pragma: no cover - defensive introspection
        return None


def _safe_type_name(value: Any) -> str:
    if isinstance(value, type):
        return value.__name__
    return type(value).__name__


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
