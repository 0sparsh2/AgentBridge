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
                "structured_output": "partial",
                "workflow.graph": "full",
                "workflow.routing": "extension",
                "workflow.roles_tasks": "extension",
                "state.session": "full",
                "state.checkpointing": "extension",
                "streaming.events": "partial",
                "human_approval": "extension",
                "observability.raw": "full",
                "agui.events": "partial",
            },
            notes={
                "workflow.graph": "LangGraph is the strongest v0 target for explicit state graphs.",
                "human_approval": "Interrupt state is reported when configured; portable approval/resume helpers are planned.",
                "agent.model": "Current adapter demonstrates graph execution without calling a model.",
                "tools.sync": "ToolSpec callables execute inside the graph node.",
                "state.checkpointing": "Enable via LangGraphExtension.config(enable_checkpointing=True).",
                "workflow.routing": "Enable via LangGraphExtension.config(route_on_context_key=..., routes=...).",
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
    ) -> list[AgentEvent]:
        events = [
            AgentEvent(
                type="workflow",
                backend=self.backend_name,
                data={"phase": "node_complete", "route": raw.get("route")},
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
