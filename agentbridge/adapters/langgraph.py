"""LangGraph adapter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypedDict

from agentbridge.adapters.base import BackendAdapter
from agentbridge.errors import MissingDependencyError
from agentbridge.extensions.langgraph import LangGraphConfig
from agentbridge.tool_execution import execute_sync_tools
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


class AgentState(TypedDict, total=False):
    input: str
    output: str
    context: dict[str, Any]
    tool_outputs: list[dict[str, Any]]
    route: str


@dataclass(frozen=True)
class LangGraphCompiledAgent:
    spec: AgentSpec
    graph: Any
    config: LangGraphConfig


class LangGraphAdapter(BackendAdapter):
    """Adapter for a minimal LangGraph state graph."""

    backend_name = "langgraph"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "partial",
                "agent.model": "partial",
                "tools.sync": "full",
                "tools.async": "partial",
                "structured_output": "full",
                "workflow.graph": "full",
                "workflow.routing": "extension",
                "workflow.roles_tasks": "extension",
                "state.session": "full",
                "state.checkpointing": "extension",
                "streaming.events": "partial",
                "human_approval": "extension",
                "observability.raw": "full",
                "observability.diagnostics": "full",
                "agui.events": "partial",
            },
            notes={
                "workflow.graph": "LangGraph is the strongest v0 target for explicit state graphs.",
                "human_approval": "Interrupt state is reported when configured and checkpointed runs can resume through resume_agent().",
                "agent.model": "Current adapter demonstrates graph execution without calling a model.",
                "tools.sync": "ToolSpec callables execute inside the graph node.",
                "structured_output": "Deterministically validates graph output against AgentSpec.output_type/output_schema; use backend_config.custom_output_args for fixture data.",
                "state.checkpointing": "Enable via LangGraphExtension.config(enable_checkpointing=True).",
                "workflow.routing": "Enable via LangGraphExtension.config(route_on_context_key=..., routes=...).",
                "observability.diagnostics": "Summarizes route, checkpointing, interrupts, tools, and normalized event counts.",
            },
        )

    def compile(self, spec: AgentSpec) -> LangGraphCompiledAgent:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "langgraph", "langgraph") from exc

        config = LangGraphConfig.model_validate(spec.backend_config.get("langgraph", {}))

        def run_node(state: AgentState, *, route_name: str) -> AgentState:
            run_input = RunInput(input=state["input"], context=state.get("context", {}))
            tool_outputs = execute_sync_tools(spec.tools, run_input)
            output = {
                "agent": spec.name,
                "input": state["input"],
                "route": route_name,
                "tools": [
                    {"name": record["name"], "result": record["result"]} for record in tool_outputs
                ],
                "message": f"{spec.name} ({spec.model}) completed LangGraph execution.",
            }
            if config.include_context_in_output:
                output["context"] = state.get("context", {})
            output = _structured_output_for_spec(
                spec,
                fallback=output,
                run_input=state["input"],
                route_name=route_name,
            )
            return {
                "input": state["input"],
                "output": output,
                "tool_outputs": tool_outputs,
                "route": route_name,
            }

        def default_node(state: AgentState) -> AgentState:
            return run_node(state, route_name=config.node_name)

        graph = StateGraph(AgentState)
        graph.add_node(config.node_name, default_node)
        route_targets = self._route_targets(config)
        for node_name in sorted(set(route_targets.values())):
            if node_name == config.node_name:
                continue

            def route_node(state: AgentState, *, _node_name: str = node_name) -> AgentState:
                return run_node(state, route_name=_node_name)

            graph.add_node(node_name, route_node)

        if route_targets:
            graph.add_node("__agentbridge_route__", lambda state: state)
            graph.set_entry_point("__agentbridge_route__")
            graph.add_conditional_edges(
                "__agentbridge_route__",
                lambda state: self._select_route(config, state),
                route_targets,
            )
        else:
            graph.set_entry_point(config.node_name)

        graph.add_edge(config.node_name, END)
        for node_name in sorted(set(route_targets.values())):
            if node_name != config.node_name:
                graph.add_edge(node_name, END)
        checkpointer = self._build_checkpointer(config)
        compiled_graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=config.interrupt_before,
            interrupt_after=config.interrupt_after,
            name=config.graph_name,
        )
        return LangGraphCompiledAgent(spec=spec, graph=compiled_graph, config=config)

    def run(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> RunResult:
        invoke_config = self._invoke_config(compiled, run_input)
        raw = compiled.graph.invoke(
            {
                "input": run_input.input,
                "context": run_input.context,
                "output": "",
                "tool_outputs": [],
                "route": "",
            },
            config=invoke_config,
        )
        interrupt_state = self._interrupt_state(compiled, invoke_config)
        output = raw.get("output", raw)
        if interrupt_state and not output:
            output = {
                "agent": compiled.spec.name,
                "input": raw.get("input", run_input.input),
                "interrupted": True,
                "next": interrupt_state["next"],
                "message": "LangGraph execution paused at an interrupt boundary.",
            }
        events = self._events_from_raw(raw, output, interrupt_state=interrupt_state)
        metadata = {
            "node_name": compiled.config.node_name,
            "checkpointing": compiled.config.enable_checkpointing,
            "route": raw.get("route", compiled.config.node_name),
            "run_diagnostics": self._run_diagnostics(raw, events, compiled, interrupt_state),
        }
        if interrupt_state:
            metadata["interrupted"] = True
            metadata["next"] = interrupt_state["next"]
            metadata["checkpoint"] = interrupt_state.get("checkpoint")
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata=metadata,
            raw=raw,
        )

    def resume(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> RunResult:
        if not compiled.config.enable_checkpointing:
            raise ValueError("LangGraph resume requires enable_checkpointing=True.")
        invoke_config = self._invoke_config(compiled, run_input)
        raw = compiled.graph.invoke(None, config=invoke_config)
        raw = raw or {}
        interrupt_state = self._interrupt_state(compiled, invoke_config)
        output = raw.get("output", raw)
        if interrupt_state and not output:
            output = {
                "agent": compiled.spec.name,
                "input": raw.get("input", run_input.input),
                "interrupted": True,
                "next": interrupt_state["next"],
                "message": "LangGraph execution paused again at an interrupt boundary.",
            }
        events = self._events_from_raw(raw, output, interrupt_state=interrupt_state, resumed=True)
        metadata = {
            "node_name": compiled.config.node_name,
            "checkpointing": compiled.config.enable_checkpointing,
            "route": raw.get("route", compiled.config.node_name),
            "resumed": True,
            "run_diagnostics": self._run_diagnostics(
                raw,
                events,
                compiled,
                interrupt_state,
                resumed=True,
            ),
        }
        if interrupt_state:
            metadata["interrupted"] = True
            metadata["next"] = interrupt_state["next"]
            metadata["checkpoint"] = interrupt_state.get("checkpoint")
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata=metadata,
            raw=raw,
        )

    def _build_checkpointer(self, config: LangGraphConfig) -> Any | None:
        if not config.enable_checkpointing:
            return None
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "langgraph", "langgraph") from exc
        return MemorySaver()

    def _invoke_config(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
    ) -> dict[str, Any] | None:
        if not compiled.config.enable_checkpointing:
            return None
        return {"configurable": {"thread_id": run_input.session_id or compiled.spec.name}}

    def _interrupt_state(
        self,
        compiled: LangGraphCompiledAgent,
        invoke_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not compiled.config.enable_checkpointing or not invoke_config:
            return None
        get_state = getattr(compiled.graph, "get_state", None)
        if not callable(get_state):
            return None
        try:
            snapshot = get_state(invoke_config)
        except Exception:  # pragma: no cover - defensive around optional LangGraph internals
            return None
        next_nodes = list(getattr(snapshot, "next", ()) or ())
        if not next_nodes:
            return None
        checkpoint = None
        snapshot_config = getattr(snapshot, "config", None)
        if isinstance(snapshot_config, dict):
            configurable = snapshot_config.get("configurable", {})
            checkpoint = {
                "thread_id": configurable.get("thread_id"),
                "checkpoint_id": configurable.get("checkpoint_id"),
                "checkpoint_ns": configurable.get("checkpoint_ns"),
            }
        return {
            "next": next_nodes,
            "checkpoint": checkpoint,
        }

    def _route_targets(self, config: LangGraphConfig) -> dict[str, str]:
        if not config.route_on_context_key:
            return {}
        targets = dict(config.routes)
        targets.setdefault("__default__", config.node_name)
        return targets

    def _select_route(self, config: LangGraphConfig, state: AgentState) -> str:
        if not config.route_on_context_key:
            return "__default__"
        context = state.get("context", {})
        route_value = str(context.get(config.route_on_context_key, "__default__"))
        if route_value in config.routes:
            return route_value
        return "__default__"

    def stream(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> Iterator[AgentEvent]:
        yield AgentEvent(
            type="workflow",
            backend=self.backend_name,
            data={
                "phase": "start",
                "node_name": compiled.config.node_name,
                "checkpointing": compiled.config.enable_checkpointing,
            },
        )
        route_targets = self._route_targets(compiled.config)
        selected_route = self._select_route(
            compiled.config,
            {"input": run_input.input, "context": run_input.context},
        )
        if route_targets:
            yield AgentEvent(
                type="workflow",
                backend=self.backend_name,
                data={
                    "phase": "route",
                    "route_key": compiled.config.route_on_context_key,
                    "route": selected_route,
                    "node": route_targets.get(selected_route),
                },
            )
        result = self.run(compiled, run_input)
        yield from result.events

    def _events_from_raw(
        self,
        raw: dict[str, Any],
        output: Any,
        *,
        interrupt_state: dict[str, Any] | None = None,
        resumed: bool = False,
    ) -> list[AgentEvent]:
        events = [
            AgentEvent(
                type="workflow",
                backend=self.backend_name,
                data={
                    "phase": "resumed" if resumed else "node_complete",
                    "route": raw.get("route"),
                },
            ),
        ]
        if interrupt_state:
            events.append(
                AgentEvent(
                    type="workflow",
                    backend=self.backend_name,
                    data={
                        "phase": "interrupted",
                        "next": interrupt_state["next"],
                        "checkpoint": interrupt_state.get("checkpoint"),
                    },
                )
            )
        events.append(
            AgentEvent(
                type="message",
                backend=self.backend_name,
                data={"role": "assistant", "content": output},
            )
        )
        for record in raw.get("tool_outputs", []):
            events.append(
                AgentEvent(
                    type="tool_call",
                    backend=self.backend_name,
                    data={"name": record["name"], "arguments": record["arguments"]},
                )
            )
            events.append(
                AgentEvent(
                    type="tool_result",
                    backend=self.backend_name,
                    data={"name": record["name"], "result": record["result"]},
                )
            )
        events.append(AgentEvent(type="complete", backend=self.backend_name, data={"output": output}))
        return events

    def _run_diagnostics(
        self,
        raw: dict[str, Any],
        events: list[AgentEvent],
        compiled: LangGraphCompiledAgent,
        interrupt_state: dict[str, Any] | None,
        *,
        resumed: bool = False,
    ) -> dict[str, Any]:
        event_counts: dict[str, int] = {}
        for event in events:
            event_counts[event.type] = event_counts.get(event.type, 0) + 1
        route_targets = self._route_targets(compiled.config)
        return {
            "route": raw.get("route", compiled.config.node_name),
            "node_name": compiled.config.node_name,
            "graph_name": compiled.config.graph_name,
            "checkpointing": compiled.config.enable_checkpointing,
            "interrupted": interrupt_state is not None,
            "resumed": resumed,
            "next": interrupt_state.get("next") if interrupt_state else [],
            "checkpoint": interrupt_state.get("checkpoint") if interrupt_state else None,
            "tool_outputs_count": len(raw.get("tool_outputs", []) or []),
            "route_targets_count": len(route_targets),
            "event_counts": event_counts,
            "structured_output": compiled.spec.output_type is not None
            or compiled.spec.output_schema is not None,
        }


def _structured_output_for_spec(
    spec: AgentSpec,
    *,
    fallback: dict[str, Any],
    run_input: str,
    route_name: str,
) -> Any:
    if spec.output_type is None and spec.output_schema is None:
        return fallback
    payload = spec.backend_config.get("custom_output_args")
    if payload is None:
        payload = _value_for_schema(
            spec.output_schema or {},
            fallback=fallback,
            run_input=run_input,
            route_name=route_name,
        )
    if spec.output_type is not None:
        model_validate = getattr(spec.output_type, "model_validate", None)
        if callable(model_validate):
            return model_validate(payload)
    return payload


def _value_for_schema(
    schema: dict[str, Any],
    *,
    fallback: dict[str, Any],
    run_input: str,
    route_name: str,
) -> Any:
    if "default" in schema:
        return schema["default"]
    raw_type = schema.get("type", "object")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    if "object" in types:
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        selected = list(required) or list(properties)
        if not selected:
            return fallback
        return {
            name: _value_for_schema(
                properties.get(name, {}),
                fallback=fallback,
                run_input=run_input,
                route_name=route_name,
            )
            for name in selected
        }
    if "boolean" in types:
        return True
    if "integer" in types:
        return 1
    if "number" in types:
        return 1.0
    if "array" in types:
        return []
    if "string" in types:
        return "ok" if route_name else run_input
    return fallback
