"""Normalized client for deployed LangGraph/Agent Server runs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
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

    def stream(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        input: Any,
    ) -> Iterator[AgentEvent]:
        for payload in self.client.stream_thread_run(
            thread_id,
            assistant_id=assistant_id,
            input=input,
        ):
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
            async for payload in native_stream(
                "POST",
                f"/threads/{thread_id}/runs/stream",
                body={"assistant_id": assistant_id, "input": input},
            ):
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
        if not events or events[-1].type != "complete":
            events.append(AgentEvent(type="complete", backend=self.backend_name, data={"output": output}))
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={"thread_id": thread_id, "assistant_id": assistant_id, "remote": True},
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
        if not events or events[-1].type != "complete":
            events.append(AgentEvent(type="complete", backend=self.backend_name, data={"output": output}))
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={"thread_id": thread_id, "assistant_id": assistant_id, "remote": True, "async": True},
        )

    def state(self, thread_id: str, *, checkpoint_id: str | None = None) -> Any:
        """Read remote graph state, including an optional checkpoint snapshot."""

        return self.client.get_thread_state(thread_id, checkpoint_id=checkpoint_id)

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
            input={"command": {"resume": resume_value}},
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
            input={"command": {"resume": resume_value}},
        )


def _normalize_remote_event(payload: Mapping[str, Any]) -> AgentEvent:
    event_name = str(payload.get("event") or payload.get("type") or "updates")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        data = {"value": data}
    normalized = event_name.lower()
    if "error" in normalized:
        event_type = "error"
    elif "tool" in normalized and "result" in normalized:
        event_type = "tool_result"
    elif "tool" in normalized:
        event_type = "tool_call"
    elif normalized in {"message", "messages", "tokens", "chunk"}:
        event_type = "message"
    elif normalized in {"complete", "end", "done"}:
        event_type = "complete"
    else:
        event_type = "workflow"
    return AgentEvent(type=event_type, backend=RemoteGraphClient.backend_name, data=data)


def _output_from_events(events: list[AgentEvent]) -> Any:
    for event in reversed(events):
        if event.type in {"message", "complete"}:
            return event.data.get("output", event.data.get("content", event.data))
    return events[-1].data if events else None


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
