"""Normalized client for deployed LangGraph/Agent Server runs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import aclosing, closing
from typing import Any

from agentbridge.types import AgentEvent, RunResult

from .langsmith_api import LangSmithAPIClient


class RemoteGraphClient:
    """Bridge LangSmith/Agent Server native runs into AgentBridge events."""

    backend_name = "langgraph_remote"

    def __init__(self, client: LangSmithAPIClient):
        self.client = client

    def create_thread(self, *, metadata: Mapping[str, Any] | None = None) -> str:
        response = self.client.create_thread(metadata=metadata)
        thread_id = response.get("thread_id") or response.get("id") if isinstance(response, dict) else None
        if not thread_id:
            raise ValueError("Remote graph thread response did not include thread_id or id.")
        return str(thread_id)

    async def acreate_thread(self, *, metadata: Mapping[str, Any] | None = None) -> str:
        """Async create for a remote graph thread."""

        response = await _async_client_call(
            self.client,
            "acreate_thread",
            "create_thread",
            metadata=metadata,
        )
        thread_id = response.get("thread_id") or response.get("id") if isinstance(response, dict) else None
        if not thread_id:
            raise ValueError("Remote graph thread response did not include thread_id or id.")
        return str(thread_id)

    def thread(self, thread_id: str) -> Any:
        """Read the native remote thread record."""

        return self.client.get_thread(thread_id)

    async def athread(self, thread_id: str) -> Any:
        """Async read for the native remote thread record."""

        return await _async_client_call(self.client, "aget_thread", "get_thread", thread_id)

    def history(self, thread_id: str, *, limit: int | None = None) -> Any:
        """Read persisted messages/checkpoints for a remote thread."""

        if limit is None:
            return self.client.get_thread_history(thread_id)
        return self.client.get_thread_history(thread_id, limit=limit)

    async def ahistory(self, thread_id: str, *, limit: int | None = None) -> Any:
        """Async read for persisted remote thread history."""

        if limit is None:
            return await _async_client_call(
                self.client, "aget_thread_history", "get_thread_history", thread_id
            )
        return await _async_client_call(
            self.client,
            "aget_thread_history",
            "get_thread_history",
            thread_id,
            limit=limit,
        )

    def copy_thread(self, thread_id: str) -> Any:
        """Copy a remote thread using the server's native persistence semantics."""

        return self.client.copy_thread(thread_id)

    async def acopy_thread(self, thread_id: str) -> Any:
        """Async copy for a remote thread."""

        return await _async_client_call(self.client, "acopy_thread", "copy_thread", thread_id)

    def search_threads(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Search remote threads using native Agent Server filters."""

        return self.client.search_threads(body=body)

    async def asearch_threads(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async search for remote threads."""

        return await _async_client_call(
            self.client,
            "asearch_threads",
            "search_threads",
            body=body,
        )

    def prune_threads(self, *, thread_ids: list[str], strategy: str) -> Any:
        """Prune remote checkpoints/threads with the server-native strategy."""

        return self.client.prune_threads(thread_ids=thread_ids, strategy=strategy)

    async def aprune_threads(self, *, thread_ids: list[str], strategy: str) -> Any:
        """Async prune for remote checkpoints/threads."""

        return await _async_client_call(
            self.client,
            "aprune_threads",
            "prune_threads",
            thread_ids=thread_ids,
            strategy=strategy,
        )

    def resolve_interrupt(self, thread_id: str) -> Any:
        """Resolve a pending remote human-in-the-loop interrupt."""

        return self.client.resolve_interrupt(thread_id)

    async def aresolve_interrupt(self, thread_id: str) -> Any:
        """Async resolution for a pending remote interrupt."""

        return await _async_client_call(
            self.client,
            "aresolve_interrupt",
            "resolve_interrupt",
            thread_id,
        )

    def assistant(self, assistant_id: str) -> Any:
        """Read a deployed assistant's native configuration."""

        return self.client.get_assistant(assistant_id)

    async def aassistant(self, assistant_id: str) -> Any:
        """Async read for a deployed assistant's configuration."""

        return await _async_client_call(self.client, "aget_assistant", "get_assistant", assistant_id)

    def assistant_graph(self, assistant_id: str, *, xray: bool | int | None = None) -> Any:
        """Read the deployed assistant graph, optionally including subgraph detail."""

        return self.client.get_assistant_graph(assistant_id, xray=xray)

    async def aassistant_graph(self, assistant_id: str, *, xray: bool | int | None = None) -> Any:
        """Async read for the deployed assistant graph."""

        return await _async_client_call(
            self.client,
            "aget_assistant_graph",
            "get_assistant_graph",
            assistant_id,
            xray=xray,
        )

    def assistant_schemas(self, assistant_id: str) -> Any:
        """Read the deployed assistant input, output, and config schemas."""

        return self.client.get_assistant_schemas(assistant_id)

    async def aassistant_schemas(self, assistant_id: str) -> Any:
        """Async read for deployed assistant schemas."""

        return await _async_client_call(
            self.client,
            "aget_assistant_schemas",
            "get_assistant_schemas",
            assistant_id,
        )

    def assistant_subgraphs(self, assistant_id: str, *, namespace: str | None = None) -> Any:
        """Read all deployed subgraphs or one native namespace."""

        return self.client.get_assistant_subgraphs(assistant_id, namespace=namespace)

    async def aassistant_subgraphs(
        self,
        assistant_id: str,
        *,
        namespace: str | None = None,
    ) -> Any:
        """Async read for deployed assistant subgraphs."""

        return await _async_client_call(
            self.client,
            "aget_assistant_subgraphs",
            "get_assistant_subgraphs",
            assistant_id,
            namespace=namespace,
        )

    def assistant_versions(
        self,
        assistant_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """List deployed assistant versions with native filters."""

        return self.client.get_assistant_versions(assistant_id, query=query)

    async def aassistant_versions(
        self,
        assistant_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list for deployed assistant versions."""

        return await _async_client_call(
            self.client,
            "aget_assistant_versions",
            "get_assistant_versions",
            assistant_id,
            query=query,
        )

    def set_latest_assistant_version(self, assistant_id: str, version: int) -> Any:
        """Select the active version for a deployed assistant."""

        return self.client.set_latest_assistant_version(assistant_id, version)

    async def aset_latest_assistant_version(self, assistant_id: str, version: int) -> Any:
        """Async select for the active deployed assistant version."""

        return await _async_client_call(
            self.client,
            "aset_latest_assistant_version",
            "set_latest_assistant_version",
            assistant_id,
            version,
        )

    def stream(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        input: Any,
    ) -> Iterator[AgentEvent]:
        with closing(self.client.stream_thread_run(
            thread_id,
            assistant_id=assistant_id,
            input=input,
        )) as events:
            for payload in events:
                yield _normalize_remote_event(payload)

    async def astream(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        input: Any,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a remote graph without blocking the application event loop."""

        native_stream = getattr(self.client, "astream_events", None)
        if callable(native_stream):
            async with aclosing(native_stream(
                "POST",
                f"/threads/{thread_id}/runs/stream",
                body={"assistant_id": assistant_id, "input": input},
            )) as events:
                async for payload in events:
                    yield _normalize_remote_event(payload)
            return

        for event in await _collect_sync_stream(
            self,
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input,
        ):
            yield event

    def run(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        input: Any,
    ) -> RunResult:
        events = list(self.stream(thread_id=thread_id, assistant_id=assistant_id, input=input))
        output = _output_from_events(events)
        failed = any(event.type == "error" for event in events)
        if not failed and (not events or events[-1].type != "complete"):
            events.append(AgentEvent(type="complete", backend=self.backend_name, data={"output": output}))
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "thread_id": thread_id, "assistant_id": assistant_id,
                "remote": True, "status": "error" if failed else "completed",
            },
        )

    async def arun(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        input: Any,
    ) -> RunResult:
        """Run a remote graph through its async event stream."""

        events = [
            event
            async for event in self.astream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=input,
            )
        ]
        output = _output_from_events(events)
        failed = any(event.type == "error" for event in events)
        if not failed and (not events or events[-1].type != "complete"):
            events.append(AgentEvent(type="complete", backend=self.backend_name, data={"output": output}))
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "thread_id": thread_id, "assistant_id": assistant_id,
                "remote": True, "async": True, "status": "error" if failed else "completed",
            },
        )

    def state(self, thread_id: str, *, checkpoint_id: str | None = None) -> Any:
        """Read remote graph state, including an optional checkpoint snapshot."""

        return self.client.get_thread_state(thread_id, checkpoint_id=checkpoint_id)

    def state_at_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        subgraphs: bool | None = None,
    ) -> Any:
        """Read state at a specific checkpoint, retaining native subgraph options."""

        return self.client.get_thread_state_at_checkpoint(
            thread_id,
            checkpoint_id,
            subgraphs=subgraphs,
        )

    async def astate_at_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        subgraphs: bool | None = None,
    ) -> Any:
        """Async read of state at a specific checkpoint."""

        return await _async_client_call(
            self.client,
            "aget_thread_state_at_checkpoint",
            "get_thread_state_at_checkpoint",
            thread_id,
            checkpoint_id,
            subgraphs=subgraphs,
        )

    async def astate(self, thread_id: str, *, checkpoint_id: str | None = None) -> Any:
        """Async read for remote graph state."""

        return await _async_client_call(
            self.client,
            "aget_thread_state",
            "get_thread_state",
            thread_id,
            checkpoint_id=checkpoint_id,
        )

    def update_state(self, thread_id: str, *, values: Any, as_node: str | None = None) -> Any:
        """Apply a native remote graph state update."""

        return self.client.update_thread_state(thread_id, values=values, as_node=as_node)

    async def aupdate_state(self, thread_id: str, *, values: Any, as_node: str | None = None) -> Any:
        """Async update for remote graph state."""

        return await _async_client_call(
            self.client,
            "aupdate_thread_state",
            "update_thread_state",
            thread_id,
            values=values,
            as_node=as_node,
        )

    def resume(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        resume_value: Any,
    ) -> Any:
        """Resume an interrupted remote run using the native command payload."""

        return self.client.create_thread_run(
            thread_id,
            assistant_id=assistant_id,
            input=None,
            command={"resume": resume_value},
        )

    async def aresume(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        resume_value: Any,
    ) -> Any:
        """Async resume for an interrupted remote run."""

        return await _async_client_call(
            self.client,
            "acreate_thread_run",
            "create_thread_run",
            thread_id,
            assistant_id=assistant_id,
            input=None,
            command={"resume": resume_value},
        )

    def run_record(self, *, thread_id: str, run_id: str) -> Any:
        """Read one persisted remote run record."""

        return self.client.get_run(thread_id, run_id)

    async def arun_record(self, *, thread_id: str, run_id: str) -> Any:
        """Async read for one persisted remote run record."""

        return await _async_client_call(
            self.client,
            "aget_run",
            "get_run",
            thread_id,
            run_id,
        )

    def run_events(self, *, thread_id: str, run_id: str) -> Any:
        """Read persisted events for one remote run."""

        return self.client.list_run_events(thread_id, run_id)

    def thread_runs(
        self,
        thread_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """List persisted runs belonging to a remote thread."""

        return self.client.list_thread_runs(thread_id, query=query)

    async def athread_runs(
        self,
        thread_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list for persisted runs belonging to a remote thread."""

        return await _async_client_call(
            self.client,
            "alist_thread_runs",
            "list_thread_runs",
            thread_id,
            query=query,
        )

    async def arun_events(self, *, thread_id: str, run_id: str) -> Any:
        """Async read for persisted events for one remote run."""

        return await _async_client_call(
            self.client,
            "alist_run_events",
            "list_run_events",
            thread_id,
            run_id,
        )

    def join_run(self, *, thread_id: str, run_id: str) -> Any:
        """Wait for and return a remote run's final native response."""

        return self.client.join_run(thread_id, run_id)

    async def ajoin_run(self, *, thread_id: str, run_id: str) -> Any:
        """Async wait for a remote run's final native response."""

        return await _async_client_call(self.client, "ajoin_run", "join_run", thread_id, run_id)

    def cancel_run(self, *, thread_id: str, run_id: str) -> Any:
        """Cancel one remote run."""

        return self.client.cancel_run(thread_id, run_id)

    async def acancel_run(self, *, thread_id: str, run_id: str) -> Any:
        """Async cancellation for one remote run."""

        return await _async_client_call(self.client, "acancel_run", "cancel_run", thread_id, run_id)

    def delete_run(self, *, thread_id: str, run_id: str) -> Any:
        """Delete one persisted remote run."""

        return self.client.delete_run(thread_id, run_id)

    async def adelete_run(self, *, thread_id: str, run_id: str) -> Any:
        """Async deletion for one persisted remote run."""

        return await _async_client_call(self.client, "adelete_run", "delete_run", thread_id, run_id)


def _normalize_remote_event(payload: Mapping[str, Any]) -> AgentEvent:
    event_name = str(payload.get("event") or payload.get("type") or "updates")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        data = {"value": data}
    normalized = event_name.lower().split("|", 1)[0]
    if "error" in normalized:
        event_type = "error"
    elif "tool" in normalized and "result" in normalized:
        event_type = "tool_result"
    elif "tool" in normalized:
        event_type = "tool_call"
    elif normalized in {"message", "messages", "messages-tuple", "tokens", "chunk"}:
        event_type = "message"
    elif normalized in {"complete", "end", "done"}:
        event_type = "complete"
    else:
        event_type = "workflow"
    metadata = {"native_event": event_name}
    if "id" in payload:
        metadata["sse_id"] = payload["id"]
    return AgentEvent(
        type=event_type, backend=RemoteGraphClient.backend_name, data=data, metadata=metadata
    )


def _output_from_events(events: list[AgentEvent]) -> Any:
    for event in reversed(events):
        if event.type == "error":
            continue
        if event.metadata.get("native_event", "").split("|", 1)[0] == "values":
            return event.data.get("value", event.data)
        if event.type in {"message", "complete"}:
            if event.data == {"value": None} or not event.data:
                continue
            return event.data.get("output", event.data.get("content", event.data))
    for event in reversed(events):
        if event.type != "error" and event.data != {"value": None}:
            return event.data
    return None


async def _collect_sync_stream(
    client: RemoteGraphClient,
    *,
    thread_id: str,
    assistant_id: str,
    input: Any,
) -> list[AgentEvent]:
    """Use the existing sync client as an explicit compatibility fallback."""

    import asyncio

    return await asyncio.to_thread(
        lambda: list(client.stream(thread_id=thread_id, assistant_id=assistant_id, input=input))
    )


async def _async_client_call(
    client: Any,
    async_name: str,
    sync_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Prefer a typed async API and retain an explicit sync compatibility fallback."""

    import asyncio

    async_method = getattr(client, async_name, None)
    if callable(async_method):
        return await async_method(*args, **kwargs)
    sync_method = getattr(client, sync_name)
    return await asyncio.to_thread(sync_method, *args, **kwargs)
