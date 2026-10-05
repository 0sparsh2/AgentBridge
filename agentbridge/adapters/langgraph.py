"""LangGraph adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
import inspect
from typing import Any, TypedDict

from agentbridge.adapters.base import BackendAdapter
from agentbridge.errors import MissingDependencyError
from agentbridge.extensions.langgraph import LangGraphConfig
from agentbridge.observability import callbacks_for_config, langsmith_context, observability_metadata
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
                "execution.batch": "partial",
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
                "observability.tracing": "extension",
                "agui.events": "partial",
            },
            notes={
                "workflow.graph": "Uses a generated graph by default or accepts a prebuilt native graph through native_graph.",
                "human_approval": "Interrupt state is reported when configured and checkpointed runs can resume through resume_agent().",
                "agent.model": "Uses a supplied native model object when configured; otherwise retains deterministic graph output.",
                "tools.sync": "ToolSpec callables execute inside the graph node.",
                "execution.batch": "Uses native graph batch/abatch when available and preserves ordered normalized results with a sequential fallback.",
                "structured_output": "Deterministically validates graph output against AgentSpec.output_type/output_schema; use backend_config.custom_output_args for fixture data.",
                "state.checkpointing": "Enable via LangGraphExtension.config(enable_checkpointing=True) or forward native compile options.",
                "state.time_travel": "Exposes native get_state, get_state_history, update_state, and checkpoint replay helpers.",
                "workflow.routing": "Enable via LangGraphExtension.config(route_on_context_key=..., routes=...).",
                "observability.diagnostics": "Summarizes route, checkpointing, interrupts, tools, and normalized event counts.",
                "observability.tracing": "Forwards callbacks, tags, metadata, and optional LangSmith/Langfuse configuration through LangGraph runtime config.",
                "observability.raw": "Preserves native_graph and forwards native_options at graph compile time.",
            },
        )

    def compile(self, spec: AgentSpec) -> LangGraphCompiledAgent:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "langgraph", "langgraph") from exc

        config = LangGraphConfig.model_validate(spec.backend_config.get("langgraph", {}))

        if config.native_graph is not None:
            return LangGraphCompiledAgent(spec=spec, graph=config.native_graph, config=config)

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
            if config.model is not None:
                output["message"] = _native_model_message(config.model, state["input"])
                output["model_type"] = type(config.model).__name__
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
        compile_options = dict(config.native_options)
        compile_options.setdefault("checkpointer", checkpointer)
        compile_options.setdefault("store", config.store)
        compile_options.setdefault("cache", config.cache)
        compile_options.setdefault("interrupt_before", config.interrupt_before)
        compile_options.setdefault("interrupt_after", config.interrupt_after)
        compile_options.setdefault("name", config.graph_name)
        compiled_graph = graph.compile(
            **compile_options,
        )
        return LangGraphCompiledAgent(spec=spec, graph=compiled_graph, config=config)

    def run(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> RunResult:
        invoke_config = self._invoke_config(compiled, run_input)
        with langsmith_context(self._observability_config(compiled)):
            raw = compiled.graph.invoke(
                {
                    "input": run_input.input,
                    "context": run_input.context,
                    "output": "",
                    "tool_outputs": [],
                    "route": "",
                },
                config=invoke_config,
                **_context_kwargs(compiled.graph.invoke, run_input.context),
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
            "deployment": _deployment_summary(compiled.config),
            "agentcore_memory_id": compiled.config.agentcore_memory_id,
            "agentcore_store_namespace": compiled.config.agentcore_store_namespace,
            "native_graph": compiled.config.native_graph is not None,
            "native_options": _safe_summary(compiled.config.native_options),
            "native_model": compiled.config.model is not None,
            "custom_checkpointer": compiled.config.checkpointer is not None,
            "custom_store": compiled.config.store is not None,
            "custom_cache": compiled.config.cache is not None,
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

    async def arun(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> RunResult:
        """Use LangGraph's native async invoke path when the graph exposes it."""

        native_ainvoke = getattr(compiled.graph, "ainvoke", None)
        if not callable(native_ainvoke):
            return await super().arun(compiled, run_input)

        invoke_config = self._invoke_config(compiled, run_input)
        with langsmith_context(self._observability_config(compiled)):
            raw = await native_ainvoke(
                {
                    "input": run_input.input,
                    "context": run_input.context,
                    "output": "",
                    "tool_outputs": [],
                    "route": "",
                },
                config=invoke_config,
                **_context_kwargs(native_ainvoke, run_input.context),
            )
        raw = raw or {}
        interrupt_state = self._interrupt_state(compiled, invoke_config)
        output = raw.get("output", raw) if isinstance(raw, dict) else raw
        if interrupt_state and not output:
            output = {
                "agent": compiled.spec.name,
                "input": raw.get("input", run_input.input),
                "interrupted": True,
                "next": interrupt_state["next"],
                "message": "LangGraph execution paused at an interrupt boundary.",
            }
        raw_mapping = raw if isinstance(raw, dict) else {"output": raw}
        events = self._events_from_raw(raw_mapping, output, interrupt_state=interrupt_state)
        metadata = {
            "node_name": compiled.config.node_name,
            "checkpointing": compiled.config.enable_checkpointing,
            "route": raw_mapping.get("route", compiled.config.node_name),
            "deployment": _deployment_summary(compiled.config),
            "agentcore_memory_id": compiled.config.agentcore_memory_id,
            "agentcore_store_namespace": compiled.config.agentcore_store_namespace,
            "native_graph": compiled.config.native_graph is not None,
            "native_options": _safe_summary(compiled.config.native_options),
            "native_model": compiled.config.model is not None,
            "custom_checkpointer": compiled.config.checkpointer is not None,
            "custom_store": compiled.config.store is not None,
            "custom_cache": compiled.config.cache is not None,
            "async": True,
            "run_diagnostics": self._run_diagnostics(raw_mapping, events, compiled, interrupt_state),
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

    def batch(
        self,
        compiled: LangGraphCompiledAgent,
        run_inputs: Iterable[RunInput],
    ) -> list[RunResult]:
        """Use LangGraph's native batch invocation when the compiled graph exposes it."""

        inputs = list(run_inputs)
        native_batch = getattr(compiled.graph, "batch", None)
        if not callable(native_batch):
            return super().batch(compiled, inputs)
        payloads = [_graph_input(run_input) for run_input in inputs]
        configs = [self._invoke_config(compiled, run_input) for run_input in inputs]
        with langsmith_context(self._observability_config(compiled)):
            raws = native_batch(payloads, config=configs)
        return [
            self._normalize_batch_result(compiled, run_input, raw, config)
            for run_input, raw, config in zip(inputs, raws, configs, strict=True)
        ]

    async def abatch(
        self,
        compiled: LangGraphCompiledAgent,
        run_inputs: Iterable[RunInput],
    ) -> list[RunResult]:
        """Use LangGraph's native async batch invocation when available."""

        inputs = list(run_inputs)
        native_abatch = getattr(compiled.graph, "abatch", None)
        if not callable(native_abatch):
            return await super().abatch(compiled, inputs)
        payloads = [_graph_input(run_input) for run_input in inputs]
        configs = [self._invoke_config(compiled, run_input) for run_input in inputs]
        with langsmith_context(self._observability_config(compiled)):
            raws = await native_abatch(payloads, config=configs)
        return [
            self._normalize_batch_result(compiled, run_input, raw, config, async_run=True)
            for run_input, raw, config in zip(inputs, raws, configs, strict=True)
        ]

    def _normalize_batch_result(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        raw: Any,
        invoke_config: dict[str, Any],
        *,
        async_run: bool = False,
    ) -> RunResult:
        raw_mapping = raw if isinstance(raw, dict) else {"output": raw}
        interrupt_state = self._interrupt_state(compiled, invoke_config)
        output = raw_mapping.get("output", raw_mapping)
        if interrupt_state and not output:
            output = {
                "agent": compiled.spec.name,
                "input": raw_mapping.get("input", run_input.input),
                "interrupted": True,
                "next": interrupt_state["next"],
                "message": "LangGraph execution paused at an interrupt boundary.",
            }
        events = self._events_from_raw(raw_mapping, output, interrupt_state=interrupt_state)
        metadata = {
            "node_name": compiled.config.node_name,
            "checkpointing": compiled.config.enable_checkpointing,
            "route": raw_mapping.get("route", compiled.config.node_name),
            "deployment": _deployment_summary(compiled.config),
            "agentcore_memory_id": compiled.config.agentcore_memory_id,
            "agentcore_store_namespace": compiled.config.agentcore_store_namespace,
            "native_graph": compiled.config.native_graph is not None,
            "native_options": _safe_summary(compiled.config.native_options),
            "native_model": compiled.config.model is not None,
            "custom_checkpointer": compiled.config.checkpointer is not None,
            "custom_store": compiled.config.store is not None,
            "custom_cache": compiled.config.cache is not None,
            "async": async_run,
            "batch": True,
            "run_diagnostics": self._run_diagnostics(raw_mapping, events, compiled, interrupt_state),
        }
        if interrupt_state:
            metadata["interrupted"] = True
            metadata["next"] = interrupt_state["next"]
            metadata["checkpoint"] = interrupt_state.get("checkpoint")
        return RunResult(output=output, backend=self.backend_name, events=events, metadata=metadata, raw=raw)

    def resume(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> RunResult:
        if not compiled.config.enable_checkpointing:
            raise ValueError("LangGraph resume requires enable_checkpointing=True.")
        invoke_config = self._invoke_config(compiled, run_input)
        with langsmith_context(self._observability_config(compiled)):
            raw = compiled.graph.invoke(
                None,
                config=invoke_config,
                **_context_kwargs(compiled.graph.invoke, run_input.context),
            )
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
            "native_graph": compiled.config.native_graph is not None,
            "native_options": _safe_summary(compiled.config.native_options),
            "native_model": compiled.config.model is not None,
            "custom_checkpointer": compiled.config.checkpointer is not None,
            "custom_store": compiled.config.store is not None,
            "custom_cache": compiled.config.cache is not None,
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

    async def aresume(self, compiled: LangGraphCompiledAgent, run_input: RunInput) -> RunResult:
        """Continue a checkpointed graph through its native async invoke path."""

        if not compiled.config.enable_checkpointing:
            raise ValueError("LangGraph resume requires enable_checkpointing=True.")
        native_ainvoke = getattr(compiled.graph, "ainvoke", None)
        if not callable(native_ainvoke):
            return await super().aresume(compiled, run_input)

        invoke_config = self._invoke_config(compiled, run_input)
        with langsmith_context(self._observability_config(compiled)):
            raw = await native_ainvoke(
                None,
                config=invoke_config,
                **_context_kwargs(native_ainvoke, run_input.context),
            )
        raw = raw or {}
        interrupt_state = self._interrupt_state(compiled, invoke_config)
        raw_mapping = raw if isinstance(raw, dict) else {"output": raw}
        output = raw_mapping.get("output", raw_mapping)
        if interrupt_state and not output:
            output = {
                "agent": compiled.spec.name,
                "input": raw_mapping.get("input", run_input.input),
                "interrupted": True,
                "next": interrupt_state["next"],
                "message": "LangGraph execution paused again at an interrupt boundary.",
            }
        events = self._events_from_raw(raw_mapping, output, interrupt_state=interrupt_state, resumed=True)
        metadata = {
            "node_name": compiled.config.node_name,
            "checkpointing": True,
            "route": raw_mapping.get("route", compiled.config.node_name),
            "resumed": True,
            "async": True,
            "native_graph": compiled.config.native_graph is not None,
            "native_options": _safe_summary(compiled.config.native_options),
            "run_diagnostics": self._run_diagnostics(
                raw_mapping,
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

    def get_state(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        checkpoint_id: str | None = None,
    ) -> Any:
        """Read the native checkpoint snapshot for a session."""

        method = getattr(compiled.graph, "get_state", None)
        if not callable(method):
            raise TypeError("The compiled LangGraph does not expose get_state().")
        return method(self._checkpoint_config(compiled, run_input, checkpoint_id=checkpoint_id))

    async def aget_state(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        checkpoint_id: str | None = None,
    ) -> Any:
        """Read a checkpoint snapshot through LangGraph's native async API when available."""

        method = getattr(compiled.graph, "aget_state", None)
        if not callable(method):
            return await asyncio.to_thread(
                self.get_state,
                compiled,
                run_input,
                checkpoint_id=checkpoint_id,
            )
        return await method(self._checkpoint_config(compiled, run_input, checkpoint_id=checkpoint_id))

    def get_state_history(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
    ) -> Any:
        """Return the native checkpoint history for a session."""

        method = getattr(compiled.graph, "get_state_history", None)
        if not callable(method):
            raise TypeError("The compiled LangGraph does not expose get_state_history().")
        return method(self._checkpoint_config(compiled, run_input))

    async def aget_state_history(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
    ) -> Any:
        """Read checkpoint history through LangGraph's native async API when available."""

        method = getattr(compiled.graph, "aget_state_history", None)
        if not callable(method):
            return await asyncio.to_thread(self.get_state_history, compiled, run_input)
        return await method(self._checkpoint_config(compiled, run_input))

    def update_state(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        values: Any,
        as_node: str | None = None,
    ) -> Any:
        """Apply a native checkpoint state update without hiding its payload."""

        method = getattr(compiled.graph, "update_state", None)
        if not callable(method):
            raise TypeError("The compiled LangGraph does not expose update_state().")
        config = self._checkpoint_config(compiled, run_input)
        if as_node is None:
            return method(config, values)
        return method(config, values, as_node=as_node)

    async def aupdate_state(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        values: Any,
        as_node: str | None = None,
    ) -> Any:
        """Apply a checkpoint update through LangGraph's native async API when available."""

        method = getattr(compiled.graph, "aupdate_state", None)
        if not callable(method):
            return await asyncio.to_thread(
                self.update_state,
                compiled,
                run_input,
                values=values,
                as_node=as_node,
            )
        config = self._checkpoint_config(compiled, run_input)
        if as_node is None:
            return await method(config, values)
        return await method(config, values, as_node=as_node)

    def replay(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        checkpoint_id: str,
    ) -> RunResult:
        """Replay a graph from a historical checkpoint using native semantics."""

        config = self._checkpoint_config(compiled, run_input, checkpoint_id=checkpoint_id)
        with langsmith_context(self._observability_config(compiled)):
            raw = compiled.graph.invoke(
                None,
                config=config,
                **_context_kwargs(compiled.graph.invoke, run_input.context),
            )
        raw = raw or {}
        output = raw.get("output", raw) if isinstance(raw, dict) else raw
        events = self._events_from_raw(raw if isinstance(raw, dict) else {"output": raw}, output)
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "node_name": compiled.config.node_name,
                "checkpointing": True,
                "replayed": True,
                "checkpoint_id": checkpoint_id,
                "native_graph": compiled.config.native_graph is not None,
                "run_diagnostics": self._run_diagnostics(
                    raw if isinstance(raw, dict) else {"output": raw},
                    events,
                    compiled,
                    None,
                ),
            },
            raw=raw,
        )

    async def areplay(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        checkpoint_id: str,
    ) -> RunResult:
        """Replay a historical checkpoint through LangGraph's native async API."""

        native_ainvoke = getattr(compiled.graph, "ainvoke", None)
        if not callable(native_ainvoke):
            return await asyncio.to_thread(
                self.replay,
                compiled,
                run_input,
                checkpoint_id=checkpoint_id,
            )
        config = self._checkpoint_config(compiled, run_input, checkpoint_id=checkpoint_id)
        with langsmith_context(self._observability_config(compiled)):
            raw = await native_ainvoke(
                None,
                config=config,
                **_context_kwargs(native_ainvoke, run_input.context),
            )
        raw = raw or {}
        raw_mapping = raw if isinstance(raw, dict) else {"output": raw}
        output = raw_mapping.get("output", raw_mapping)
        events = self._events_from_raw(raw_mapping, output)
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "node_name": compiled.config.node_name,
                "checkpointing": True,
                "replayed": True,
                "async": True,
                "checkpoint_id": checkpoint_id,
                "native_graph": compiled.config.native_graph is not None,
                "run_diagnostics": self._run_diagnostics(
                    raw_mapping,
                    events,
                    compiled,
                    None,
                ),
            },
            raw=raw,
        )

    def _build_checkpointer(self, config: LangGraphConfig) -> Any | None:
        if config.checkpointer is not None:
            return config.checkpointer
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
        config = self._observability_config(compiled)
        invoke_config: dict[str, Any] = {}
        if compiled.config.enable_checkpointing or compiled.config.checkpointer is not None:
            invoke_config["configurable"] = {
                "thread_id": run_input.session_id or compiled.spec.name
            }
        callbacks = callbacks_for_config(config)
        if callbacks:
            invoke_config["callbacks"] = callbacks
        metadata = observability_metadata(
            {
                "metadata": compiled.config.metadata,
                "observability": compiled.config.observability,
            },
            metadata=run_input.metadata,
            session_id=run_input.session_id,
        )
        if metadata:
            invoke_config["metadata"] = metadata
        observability = compiled.config.observability
        if observability.get("tags"):
            invoke_config["tags"] = list(observability["tags"])
        if observability.get("run_name"):
            invoke_config["run_name"] = observability["run_name"]
        return invoke_config or None

    def _checkpoint_config(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
        *,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a state API config without dropping observability metadata."""

        config = self._invoke_config(compiled, run_input) or {}
        configurable = dict(config.get("configurable", {}))
        configurable.setdefault("thread_id", run_input.session_id or compiled.spec.name)
        if checkpoint_id is not None:
            configurable["checkpoint_id"] = checkpoint_id
        config["configurable"] = configurable
        return config

    def _observability_config(self, compiled: LangGraphCompiledAgent) -> dict[str, Any]:
        return {
            "callbacks": compiled.config.callbacks,
            "metadata": compiled.config.metadata,
            "observability": compiled.config.observability,
        }

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
        native_stream = getattr(compiled.graph, "stream", None)
        if not callable(native_stream):
            result = self.run(compiled, run_input)
            yield from result.events
            return

        payload = {
            "input": run_input.input,
            "context": run_input.context,
            "output": "",
            "tool_outputs": [],
            "route": "",
        }
        invoke_config = self._invoke_config(compiled, run_input)
        yielded = False
        last_output: Any = None
        with langsmith_context(self._observability_config(compiled)):
            try:
                stream = native_stream(
                    payload,
                    config=invoke_config,
                    **self._stream_options(compiled),
                    **_context_kwargs(native_stream, run_input.context),
                )
            except TypeError:
                stream = native_stream(payload, config=invoke_config)
            for chunk in stream:
                yielded = True
                events, output = _events_from_native_graph_chunk(chunk, self.backend_name)
                if output is not None:
                    last_output = output
                yield from events
        if not yielded:
            result = self.run(compiled, run_input)
            yield from result.events
            return
        yield AgentEvent(
            type="complete",
            backend=self.backend_name,
            data={"output": last_output},
        )

    async def astream(
        self,
        compiled: LangGraphCompiledAgent,
        run_input: RunInput,
    ) -> AsyncIterator[AgentEvent]:
        """Consume LangGraph's native async stream when supported by the graph."""

        native_astream = getattr(compiled.graph, "astream", None)
        if not callable(native_astream):
            async for event in super().astream(compiled, run_input):
                yield event
            return

        payload = {
            "input": run_input.input,
            "context": run_input.context,
            "output": "",
            "tool_outputs": [],
            "route": "",
        }
        invoke_config = self._invoke_config(compiled, run_input)
        yielded = False
        last_output: Any = None
        with langsmith_context(self._observability_config(compiled)):
            try:
                stream = native_astream(
                    payload,
                    config=invoke_config,
                    **self._stream_options(compiled),
                    **_context_kwargs(native_astream, run_input.context),
                )
                async for chunk in stream:
                    yielded = True
                    events, output = _events_from_native_graph_chunk(chunk, self.backend_name)
                    if output is not None:
                        last_output = output
                    for event in events:
                        yield event
            except TypeError:
                stream = native_astream(payload, config=invoke_config)
                async for chunk in stream:
                    yielded = True
                    events, output = _events_from_native_graph_chunk(chunk, self.backend_name)
                    if output is not None:
                        last_output = output
                    for event in events:
                        yield event

        if not yielded:
            result = await super().arun(compiled, run_input)
            for event in result.events:
                yield event
            return
        yield AgentEvent(
            type="complete",
            backend=self.backend_name,
            data={"output": last_output},
        )

    def _stream_options(self, compiled: LangGraphCompiledAgent) -> dict[str, Any]:
        """Forward supported native stream options while keeping stable defaults."""

        native_options = compiled.config.native_options
        options: dict[str, Any] = {
            "stream_mode": native_options.get("stream_mode", ["updates", "messages", "custom"]),
            "version": native_options.get("version", "v2"),
        }
        for name in ("subgraphs", "print_mode", "output_keys", "durability"):
            if name in native_options:
                options[name] = native_options[name]
        return options

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
        events.append(
            AgentEvent(type="complete", backend=self.backend_name, data={"output": output})
        )
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
            "deployment": _deployment_summary(compiled.config),
            "agentcore_memory_id": compiled.config.agentcore_memory_id,
            "agentcore_store_namespace": compiled.config.agentcore_store_namespace,
            "native_graph": compiled.config.native_graph is not None,
            "native_options": _safe_summary(compiled.config.native_options),
            "native_model": compiled.config.model is not None,
            "custom_checkpointer": compiled.config.checkpointer is not None,
            "custom_store": compiled.config.store is not None,
            "custom_cache": compiled.config.cache is not None,
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


def _graph_input(run_input: RunInput) -> dict[str, Any]:
    """Build the portable state payload used by generated and native graphs."""

    return {
        "input": run_input.input,
        "context": run_input.context,
        "output": "",
        "tool_outputs": [],
        "route": "",
    }


def _events_from_native_graph_chunk(
    chunk: Any,
    backend: str,
) -> tuple[list[AgentEvent], Any | None]:
    """Normalize LangGraph updates/messages without requiring a second graph run."""

    namespace: Any = None
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[1], dict):
        namespace, chunk = chunk
    if not isinstance(chunk, dict):
        return [
            AgentEvent(
                type="workflow",
                backend=backend,
                data={
                    "value": chunk,
                    "native_type": type(chunk).__name__,
                    **({"namespace": namespace} if namespace is not None else {}),
                },
            )
        ], None

    events: list[AgentEvent] = []
    output = None
    for source, update in chunk.items():
        if source in {"messages", "__root__"} and isinstance(update, (list, tuple)):
            for message in update:
                content = message.get("content") if isinstance(message, dict) else getattr(message, "content", message)
                events.append(
                    AgentEvent(
                        type="message",
                        backend=backend,
                        data={
                            "content": content,
                            "source": str(source),
                            **({"namespace": namespace} if namespace is not None else {}),
                        },
                    )
                )
            continue
        if not isinstance(update, dict):
            events.append(
                AgentEvent(
                    type="workflow",
                    backend=backend,
                    data={
                        "source": str(source),
                        "value": update,
                        **({"namespace": namespace} if namespace is not None else {}),
                    },
                )
            )
            continue
        if update.get("output") is not None:
            output = update["output"]
            events.append(
                AgentEvent(
                    type="message",
                    backend=backend,
                    data={
                        "role": "assistant",
                        "content": output,
                        "source": str(source),
                        **({"namespace": namespace} if namespace is not None else {}),
                    },
                )
            )
        for record in update.get("tool_outputs", []) or []:
            events.append(
                AgentEvent(
                    type="tool_result",
                    backend=backend,
                    data={
                        "name": record.get("name"),
                        "result": record.get("result"),
                        **({"namespace": namespace} if namespace is not None else {}),
                    },
                )
            )
        events.append(
            AgentEvent(
                type="workflow",
                backend=backend,
                data={
                    "phase": "update",
                    "source": str(source),
                    "update": update,
                    **({"namespace": namespace} if namespace is not None else {}),
                },
            )
        )
    return events, output


def _context_kwargs(callable_object: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Pass runtime context only when the native callable exposes that channel."""

    if not context:
        return {}
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return {"context": context}
    if any(parameter.name == "context" for parameter in parameters):
        return {"context": context}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return {"context": context}
    return {}


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


def _deployment_summary(config: LangGraphConfig) -> dict[str, Any]:
    """Expose remote/deployment intent without making a hosted API call."""

    summary = dict(config.deployment)
    if config.remote_graph:
        summary["remote_graph"] = config.remote_graph
    if config.deployment_url:
        summary["deployment_url"] = config.deployment_url
    return summary


def _safe_summary(value: Any) -> Any:
    """Summarize native objects without leaking reprs or secrets into diagnostics."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_summary(item) for item in value]
    return {"type": type(value).__name__}


def _native_model_message(model: Any, user_input: str) -> str:
    """Invoke a supplied LangChain-compatible model without importing a provider package."""

    invoke = getattr(model, "invoke", None)
    if not callable(invoke):
        raise TypeError("LangGraph native model must expose invoke().")
    response = invoke([{"role": "user", "content": user_input}])
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is not None:
        return _content_text(content)
    if isinstance(response, dict) and "content" in response:
        return _content_text(response["content"])
    return str(response)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", item.get("content", item)))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content)


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
