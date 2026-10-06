"""Small dependency-free transport for LangSmith deployment/API endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Mapping[str, str], bytes]]


class LangSmithAPIClient:
    """Authenticated JSON/SSE client with an escape hatch for every endpoint.

    This client intentionally does not mirror hundreds of hosted API methods.
    The generated/openapi surfaces change independently of AgentBridge. The
    `request_json` and `stream_events` methods preserve arbitrary paths,
    query parameters, request bodies, and response payloads while keeping
    credentials outside AgentSpec and the core SDK.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.smith.langchain.com",
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("LANGSMITH_API_KEY or api_key is required for LangSmith API calls.")
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _default_transport

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Call any LangSmith JSON endpoint and return its decoded payload."""

        status, response_headers, raw = self._request(
            method,
            path,
            query=query,
            body=body,
            headers=headers,
        )
        if status >= 400:
            detail = raw.decode("utf-8", errors="replace")
            raise RuntimeError(f"LangSmith API {status} for {method} {path}: {detail}")
        if not raw:
            return None
        content_type = response_headers.get("content-type", "")
        if "json" not in content_type and not raw.lstrip().startswith((b"{", b"[")):
            return raw
        return json.loads(raw)

    def call(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> Any:
        """Call any LangSmith JSON or SSE endpoint through one stable entry point."""

        if stream:
            return self.stream_events(
                method,
                path,
                query=query,
                body=body,
                headers=headers,
            )
        return self.request_json(
            method,
            path,
            query=query,
            body=body,
            headers=headers,
        )

    def stream_events(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Parse newline-delimited SSE data from a streaming endpoint."""

        status, _response_headers, raw = self._request(
            method,
            path,
            query=query,
            body=body,
            headers={"Accept": "text/event-stream", **(headers or {})},
        )
        if status >= 400:
            raise RuntimeError(f"LangSmith API {status} for {method} {path}")
        for line in raw.decode("utf-8").splitlines():
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload and payload != "[DONE]":
                yield json.loads(payload)

    async def arequest_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Async facade for arbitrary LangSmith JSON endpoints."""

        return await asyncio.to_thread(
            self.request_json,
            method,
            path,
            query=query,
            body=body,
            headers=headers,
        )

    async def acall(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> Any:
        """Async entry point for arbitrary LangSmith JSON or SSE endpoints."""

        if stream:
            return self.astream_events(
                method,
                path,
                query=query,
                body=body,
                headers=headers,
            )
        return await self.arequest_json(
            method,
            path,
            query=query,
            body=body,
            headers=headers,
        )

    async def astream_events(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async-compatible SSE facade preserving the dependency-free transport."""

        events = await asyncio.to_thread(
            lambda: list(
                self.stream_events(
                    method,
                    path,
                    query=query,
                    body=body,
                    headers=headers,
                )
            )
        )
        for event in events:
            yield event

    async def aget_thread(self, thread_id: str) -> Any:
        """Async fetch for a deployment thread."""

        return await self.arequest_json("GET", f"/threads/{thread_id}")

    async def acreate_thread(self, *, metadata: Mapping[str, Any] | None = None) -> Any:
        """Async create for a deployment thread."""

        return await self.arequest_json(
            "POST",
            "/threads",
            body={"metadata": dict(metadata or {})},
        )

    async def adelete_thread(self, thread_id: str) -> Any:
        """Async deletion for a deployment thread."""

        return await self.arequest_json("DELETE", f"/threads/{thread_id}")

    async def athread_history(self, thread_id: str, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async read for a deployment thread history."""

        return await self.arequest_json("POST", f"/threads/{thread_id}/history", body=dict(body or {}))

    async def aget_thread_history(self, thread_id: str, *, limit: int | None = None) -> Any:
        """Async GET convenience read for thread history."""

        return await self.arequest_json("GET", f"/threads/{thread_id}/history", query={"limit": limit})

    async def ajoin_thread_stream(
        self,
        thread_id: str,
        *,
        stream_modes: str | list[str] | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async subscribe to the SSE stream for all runs on a thread."""

        query: dict[str, Any] = {}
        if stream_modes is not None:
            query["stream_modes"] = stream_modes
        headers = {"Last-Event-ID": last_event_id} if last_event_id is not None else None
        async for event in self.astream_events(
            "GET",
            f"/threads/{thread_id}/stream",
            query=query,
            headers=headers,
        ):
            yield event

    async def aresolve_interrupt(self, thread_id: str) -> Any:
        """Async resolve for a paused human-in-the-loop thread."""

        return await self.arequest_json("POST", f"/threads/{thread_id}/resolve-interrupt")

    async def aprune_threads(self, *, thread_ids: list[str], strategy: str) -> Any:
        """Async prune for thread checkpoints or threads."""

        return await self.arequest_json(
            "POST",
            "/threads/prune",
            body={"thread_ids": thread_ids, "strategy": strategy},
        )

    async def asearch_threads(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async thread search using native Agent Server filters."""

        return await self.arequest_json("POST", "/threads/search", body=dict(body or {}))

    async def acount_threads(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async count for deployment threads matching native filters."""

        return await self.arequest_json("POST", "/threads/count", body=dict(body or {}))

    async def apatch_thread(self, thread_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async patch for deployment thread metadata."""

        return await self.arequest_json("PATCH", f"/threads/{thread_id}", body=dict(body))

    async def acopy_thread(self, thread_id: str) -> Any:
        """Async copy for a deployment thread and its persisted state."""

        return await self.arequest_json("POST", f"/threads/{thread_id}/copy")

    async def aget_assistant(self, assistant_id: str) -> Any:
        """Async fetch for a deployed assistant configuration."""

        return await self.arequest_json("GET", f"/assistants/{assistant_id}")

    async def acreate_assistant(
        self,
        *,
        graph_id: str,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async create for a deployed assistant configuration."""

        return await self.arequest_json(
            "POST",
            "/assistants",
            body={"graph_id": graph_id, "config": dict(config or {})},
        )

    async def aupdate_assistant(self, assistant_id: str, *, config: Mapping[str, Any]) -> Any:
        """Async update for an assistant's full native configuration."""

        return await self.arequest_json(
            "PATCH",
            f"/assistants/{assistant_id}",
            body={"config": dict(config)},
        )

    async def adelete_assistant(self, assistant_id: str) -> Any:
        """Async delete for an assistant and its versions."""

        return await self.arequest_json("DELETE", f"/assistants/{assistant_id}")

    async def asearch_assistants(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async assistant search using native request filters."""

        return await self.arequest_json("POST", "/assistants/search", body=dict(body or {}))

    async def acount_assistants(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async count for assistants matching native search filters."""

        return await self.arequest_json("POST", "/assistants/count", body=dict(body or {}))

    async def aget_assistant_graph(self, assistant_id: str, *, xray: bool | int | None = None) -> Any:
        """Async fetch for an assistant graph definition."""

        query = {"xray": xray} if xray is not None else None
        return await self.arequest_json("GET", f"/assistants/{assistant_id}/graph", query=query)

    async def aget_assistant_schemas(self, assistant_id: str) -> Any:
        """Async fetch for assistant input, output, and config schemas."""

        return await self.arequest_json("GET", f"/assistants/{assistant_id}/schemas")

    async def aget_assistant_subgraphs(
        self,
        assistant_id: str,
        *,
        namespace: str | None = None,
    ) -> Any:
        """Async fetch for all assistant subgraphs or one namespace."""

        path = f"/assistants/{assistant_id}/subgraphs"
        if namespace is not None:
            path = f"{path}/{quote(namespace, safe='')}"
        return await self.arequest_json("GET", path)

    async def aget_assistant_versions(
        self,
        assistant_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list for all versions of an assistant."""

        return await self.arequest_json("GET", f"/assistants/{assistant_id}/versions", query=query)

    async def aset_latest_assistant_version(self, assistant_id: str, version: int) -> Any:
        """Async select for the active assistant version."""

        return await self.arequest_json(
            "POST",
            f"/assistants/{assistant_id}/latest",
            body={"version": version},
        )

    async def aget_thread_state(
        self,
        thread_id: str,
        *,
        checkpoint_id: str | None = None,
    ) -> Any:
        """Async read of remote graph state."""

        return await self.arequest_json(
            "GET",
            f"/threads/{thread_id}/state",
            query={"checkpoint_id": checkpoint_id},
        )

    async def aget_thread_state_at_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        subgraphs: bool | None = None,
    ) -> Any:
        """Async read for state at a specific checkpoint."""

        return await self.arequest_json(
            "GET",
            f"/threads/{thread_id}/state/{checkpoint_id}",
            query={"subgraphs": subgraphs},
        )

    async def aget_thread_state_at_checkpoint_body(
        self,
        thread_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Async checkpoint-state read using the native POST payload."""

        return await self.arequest_json("POST", f"/threads/{thread_id}/state/checkpoint", body=dict(body))

    async def aupdate_thread_state(
        self,
        thread_id: str,
        *,
        values: Any,
        as_node: str | None = None,
    ) -> Any:
        """Async remote graph state update."""

        body: dict[str, Any] = {"values": values}
        if as_node is not None:
            body["as_node"] = as_node
        return await self.arequest_json("POST", f"/threads/{thread_id}/state", body=body)

    async def acreate_thread_run(
        self,
        thread_id: str,
        *,
        assistant_id: str,
        input: Any,
        stream: bool = False,
        command: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async start of a non-streaming or SSE thread run."""

        body = {"assistant_id": assistant_id, "input": input}
        if command is not None:
            body["command"] = dict(command)
        if stream:
            return self.astream_events("POST", f"/threads/{thread_id}/runs/stream", body=body)
        return await self.arequest_json("POST", f"/threads/{thread_id}/runs", body=body)

    async def aget_run(self, thread_id: str, run_id: str) -> Any:
        """Async fetch for one thread run."""

        return await self.arequest_json("GET", f"/threads/{thread_id}/runs/{run_id}")

    async def adelete_run(self, thread_id: str, run_id: str) -> Any:
        """Async deletion for one persisted thread run."""

        return await self.arequest_json("DELETE", f"/threads/{thread_id}/runs/{run_id}")

    async def alist_thread_runs(
        self,
        thread_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list for runs belonging to one thread."""

        return await self.arequest_json("GET", f"/threads/{thread_id}/runs", query=query)

    async def alist_run_events(self, thread_id: str, run_id: str) -> Any:
        """Async fetch for the persisted event list of one run."""

        return await self.arequest_json("GET", f"/threads/{thread_id}/runs/{run_id}/events")

    async def ajoin_run(self, thread_id: str, run_id: str) -> Any:
        """Async wait for the final result of one thread run."""

        return await self.arequest_json("GET", f"/threads/{thread_id}/runs/{run_id}/join")

    async def ajoin_run_stream(self, thread_id: str, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Async stream for the final result of one thread run."""

        async for event in self.astream_events("GET", f"/threads/{thread_id}/runs/{run_id}/join"):
            yield event

    async def acancel_runs(
        self,
        thread_id: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async cancel for one or more active thread runs."""

        return await self.arequest_json(
            "POST",
            f"/threads/{thread_id}/runs/cancel",
            body=dict(body or {}),
        )

    async def acreate_dataset(
        self,
        *,
        name: str,
        description: str | None = None,
        data_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async create for a LangSmith dataset."""

        body: dict[str, Any] = {"name": name}
        optional = {
            "description": description,
            "data_type": data_type,
            "metadata": dict(metadata) if metadata is not None else None,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return await self.arequest_json("POST", "/api/v1/datasets", body=body)

    async def alist_datasets(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async list for LangSmith datasets."""

        return await self.arequest_json("GET", "/api/v1/datasets", query=query)

    async def aiter_datasets(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for LangSmith datasets."""

        page_query = dict(query or {})
        while True:
            page = await self.alist_datasets(query=page_query)
            if not isinstance(page, Mapping):
                return
            for item in page.get("datasets", page.get("items", [])) or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor") or page.get("cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    async def aget_dataset(self, dataset_id: str) -> Any:
        """Async fetch for one LangSmith dataset."""

        return await self.arequest_json("GET", f"/api/v1/datasets/{dataset_id}")

    async def aupdate_dataset(self, dataset_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async update for dataset metadata."""

        return await self.arequest_json("PATCH", f"/api/v1/datasets/{dataset_id}", body=dict(body))

    async def adelete_dataset(self, dataset_id: str) -> Any:
        """Async deletion for one dataset and its examples."""

        return await self.arequest_json("DELETE", f"/api/v1/datasets/{dataset_id}")

    async def aread_dataset_version(
        self,
        dataset_id: str,
        *,
        as_of: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """Async fetch a dataset version by timestamp or tag."""

        if (as_of is None) == (tag is None):
            raise ValueError("Exactly one of as_of and tag must be specified.")
        query = {"as_of": as_of} if as_of is not None else {"tag": tag}
        return await self.arequest_json(
            "GET", f"/api/v1/datasets/{dataset_id}/version", query=query
        )

    async def alist_dataset_versions(
        self,
        dataset_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list version history for a dataset."""

        return await self.arequest_json(
            "GET", f"/api/v1/datasets/{dataset_id}/versions", query=query
        )

    async def adiff_dataset_versions(
        self,
        dataset_id: str,
        *,
        from_version: str,
        to_version: str,
    ) -> Any:
        """Async return additions/removals between two dataset versions."""

        return await self.arequest_json(
            "GET",
            f"/api/v1/datasets/{dataset_id}/versions/diff",
            query={"from_version": from_version, "to_version": to_version},
        )

    async def alist_dataset_splits(
        self,
        dataset_id: str,
        *,
        as_of: str | None = None,
    ) -> Any:
        """Async list split names for a dataset version."""

        query = {"as_of": as_of} if as_of is not None else None
        return await self.arequest_json(
            "GET", f"/api/v1/datasets/{dataset_id}/splits", query=query
        )

    async def aupdate_dataset_splits(
        self,
        dataset_id: str,
        *,
        split_name: str,
        example_ids: list[str],
        remove: bool = False,
    ) -> Any:
        """Async add or remove examples from a dataset split."""

        return await self.arequest_json(
            "PUT",
            f"/api/v1/datasets/{dataset_id}/splits",
            body={
                "split_name": split_name,
                "examples": list(example_ids),
                "remove": remove,
            },
        )

    async def ashare_dataset(self, dataset_id: str) -> Any:
        """Async create or refresh a public share for a dataset."""

        return await self.arequest_json(
            "PUT",
            f"/api/v1/datasets/{dataset_id}/share",
            body={"dataset_id": dataset_id},
        )

    async def aunshare_dataset(self, dataset_id: str) -> Any:
        """Async remove the public share for a dataset."""

        return await self.arequest_json("DELETE", f"/api/v1/datasets/{dataset_id}/share")

    async def aread_dataset_delta(
        self,
        dataset_id: str,
        *,
        baseline_session_id: str,
        comparison_session_ids: list[str],
        feedback_key: str,
        filters: Mapping[str, Any] | None = None,
        offset: int = 0,
        limit: int = 100,
        comparative_experiment_id: str | None = None,
    ) -> Any:
        """Async compare feedback regressions and improvements across sessions."""

        body: dict[str, Any] = {
            "baseline_session_id": baseline_session_id,
            "comparison_session_ids": list(comparison_session_ids),
            "feedback_key": feedback_key,
            "offset": offset,
            "limit": limit,
        }
        if filters is not None:
            body["filters"] = dict(filters)
        if comparative_experiment_id is not None:
            body["comparative_experiment_id"] = comparative_experiment_id
        return await self.arequest_json(
            "POST", f"/api/v1/datasets/{dataset_id}/runs/delta", body=body
        )

    async def aread_shared_dataset_examples_with_runs(
        self,
        share_token: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Async read examples and associated runs from a shared dataset."""

        return await self.arequest_json(
            "POST",
            f"/api/v1/public/{share_token}/examples/runs",
            body=dict(body),
        )

    async def acreate_example(
        self,
        *,
        dataset_id: str,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        example_id: str | None = None,
        name: str | None = None,
    ) -> Any:
        """Async create for one LangSmith dataset example."""

        body: dict[str, Any] = {"dataset_id": dataset_id, "inputs": dict(inputs)}
        optional = {
            "outputs": dict(outputs) if outputs is not None else None,
            "metadata": dict(metadata) if metadata is not None else None,
            "id": example_id,
            "name": name,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return await self.arequest_json("POST", "/api/v1/examples", body=body)

    async def alist_examples(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async list for LangSmith dataset examples."""

        return await self.arequest_json("GET", "/api/v1/examples", query=query)

    async def aiter_examples(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for LangSmith examples."""

        page_query = dict(query or {})
        while True:
            page = await self.alist_examples(query=page_query)
            if not isinstance(page, Mapping):
                return
            for item in page.get("examples", page.get("items", [])) or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor") or page.get("cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    async def aget_example(self, example_id: str) -> Any:
        """Async fetch for one LangSmith dataset example."""

        return await self.arequest_json("GET", f"/api/v1/examples/{example_id}")

    async def aupdate_example(self, example_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async update for one dataset example."""

        return await self.arequest_json("PATCH", f"/api/v1/examples/{example_id}", body=dict(body))

    async def adelete_example(self, example_id: str) -> Any:
        """Async deletion for one dataset example."""

        return await self.arequest_json("DELETE", f"/api/v1/examples/{example_id}")

    async def adelete_examples(
        self,
        example_ids: list[str],
        *,
        hard_delete: bool = False,
    ) -> Any:
        """Async delete multiple examples, optionally using the hard-delete endpoint."""

        if hard_delete:
            return await self.arequest_json(
                "POST",
                "/api/v1/platform/datasets/examples/delete",
                body={"example_ids": list(example_ids), "hard_delete": True},
            )
        return await self.arequest_json(
            "DELETE", "/api/v1/examples", query={"example_ids": list(example_ids)}
        )

    async def alist_shared_examples(
        self,
        share_token: str,
        *,
        example_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> Any:
        """Async list examples from a public dataset share."""

        query: dict[str, Any] = {}
        if example_ids is not None:
            query["id"] = list(example_ids)
        if limit is not None:
            query["limit"] = limit
        return await self.arequest_json(
            "GET", f"/api/v1/public/{share_token}/examples", query=query or None
        )

    async def aread_dataset_openai_finetuning(self, dataset_id: str) -> Any:
        """Async download a dataset in OpenAI fine-tuning JSONL format."""

        return await self.arequest_json("GET", f"/api/v1/datasets/{dataset_id}/openai_ft")

    async def alist_feedback(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async list for LangSmith feedback records."""

        return await self.arequest_json("GET", "/api/v1/feedback", query=query)

    async def aget_feedback(self, feedback_id: str) -> Any:
        """Async fetch for one feedback record."""

        return await self.arequest_json("GET", f"/api/v1/feedback/{feedback_id}")

    async def aupdate_feedback(self, feedback_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async update for one feedback record."""

        return await self.arequest_json("PATCH", f"/api/v1/feedback/{feedback_id}", body=dict(body))

    async def adelete_feedback(self, feedback_id: str) -> Any:
        """Async deletion for one feedback record."""

        return await self.arequest_json("DELETE", f"/api/v1/feedback/{feedback_id}")

    async def acreate_feedback_config(
        self,
        feedback_key: str,
        *,
        feedback_config: Mapping[str, Any],
        is_lower_score_better: bool = False,
    ) -> Any:
        """Async create a LangSmith feedback configuration."""

        return await self.arequest_json(
            "POST",
            "/api/v1/feedback-configs",
            body={
                "feedback_key": feedback_key,
                "feedback_config": dict(feedback_config),
                "is_lower_score_better": is_lower_score_better,
            },
        )

    async def alist_feedback_configs(
        self, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """Async list LangSmith feedback configurations."""

        return await self.arequest_json("GET", "/api/v1/feedback-configs", query=query)

    async def aupdate_feedback_config(
        self, feedback_key: str, *, body: Mapping[str, Any]
    ) -> Any:
        """Async update a LangSmith feedback configuration."""

        payload = {"feedback_key": feedback_key, **dict(body)}
        return await self.arequest_json("PATCH", "/api/v1/feedback-configs", body=payload)

    async def adelete_feedback_config(self, feedback_key: str) -> Any:
        """Async soft-delete a LangSmith feedback configuration."""

        return await self.arequest_json(
            "DELETE", "/api/v1/feedback-configs", query={"feedback_key": feedback_key}
        )

    async def acreate_presigned_feedback_token(
        self,
        run_id: str,
        feedback_key: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async create a browser-safe presigned feedback ingest token."""

        payload = {"run_id": run_id, "feedback_key": feedback_key, **dict(body or {})}
        return await self.arequest_json("POST", "/api/v1/feedback/tokens", body=payload)

    async def alist_presigned_feedback_tokens(
        self, run_id: str, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """Async list presigned feedback tokens for a run."""

        payload = {"run_id": run_id, **dict(query or {})}
        return await self.arequest_json("GET", "/api/v1/feedback/tokens", query=payload)

    async def acreate_agent_connection(
        self,
        agent_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Async create for an Agent Auth connection."""

        return await self.arequest_json(
            "POST",
            f"/v2/auth/agents/{agent_id}/connections",
            body=dict(body),
        )

    async def alist_agent_connections(self, agent_id: str) -> Any:
        """Async list for Agent Auth connections."""

        return await self.arequest_json("GET", f"/v2/auth/agents/{agent_id}/connections")

    async def aremove_agent_connection(self, agent_id: str, connection_id: str) -> Any:
        """Async removal for one Agent Auth connection."""

        return await self.arequest_json(
            "DELETE",
            f"/v2/auth/agents/{agent_id}/connections/{connection_id}",
        )

    async def alist_connection_tokens(
        self, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """Async list for Fleet Agent Auth connection tokens."""

        return await self.arequest_json("GET", "/v1/fleet/auth-tokens", query=query)

    async def aupdate_connection_token(
        self, token_id: str, *, body: Mapping[str, Any]
    ) -> Any:
        """Async update for connection-token metadata such as label/default."""

        return await self.arequest_json(
            "PATCH", f"/v1/fleet/auth-tokens/{token_id}", body=dict(body)
        )

    async def arevoke_connection_token(self, token_id: str) -> Any:
        """Async revoke for one Fleet Agent Auth connection token."""

        return await self.arequest_json("DELETE", f"/v1/fleet/auth-tokens/{token_id}")

    async def acreate_annotation_queue(
        self,
        *,
        name: str,
        description: str | None = None,
        queue_id: str | None = None,
        rubric_instructions: str | None = None,
        rubric_items: list[Mapping[str, Any]] | None = None,
    ) -> Any:
        """Async create a LangSmith human-review annotation queue."""

        body: dict[str, Any] = {"name": name}
        optional = {
            "description": description,
            "id": queue_id,
            "rubric_instructions": rubric_instructions,
            "rubric_items": rubric_items,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return await self.arequest_json("POST", "/api/v1/annotation-queues", body=body)

    async def alist_annotation_queues(
        self, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """Async list LangSmith annotation queues."""

        return await self.arequest_json("GET", "/api/v1/annotation-queues", query=query)

    async def aget_annotation_queue(self, queue_id: str) -> Any:
        """Async fetch one annotation queue."""

        return await self.arequest_json("GET", f"/api/v1/annotation-queues/{queue_id}")

    async def aupdate_annotation_queue(
        self, queue_id: str, *, body: Mapping[str, Any]
    ) -> Any:
        """Async update annotation queue metadata and rubric."""

        return await self.arequest_json(
            "PATCH", f"/api/v1/annotation-queues/{queue_id}", body=dict(body)
        )

    async def adelete_annotation_queue(self, queue_id: str) -> Any:
        """Async delete one annotation queue."""

        return await self.arequest_json("DELETE", f"/api/v1/annotation-queues/{queue_id}")

    async def aadd_runs_to_annotation_queue(
        self,
        queue_id: str,
        *,
        run_ids: list[str] | None = None,
        runs: list[Mapping[str, Any]] | None = None,
    ) -> Any:
        """Async add run IDs or full run keys to an annotation queue."""

        if (run_ids is None) == (runs is None):
            raise ValueError("Provide exactly one of run_ids or runs.")
        path = (
            f"/api/v1/annotation-queues/{queue_id}/runs"
            if run_ids is not None
            else f"/api/v1/annotation-queues/{queue_id}/runs/by-key"
        )
        payload: Any = list(run_ids) if run_ids is not None else [dict(run) for run in runs or []]
        return await self.arequest_json("POST", path, body=payload)

    async def alist_annotation_queue_runs(
        self, queue_id: str, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """Async list runs assigned to an annotation queue."""

        return await self.arequest_json(
            "GET", f"/api/v1/annotation-queues/{queue_id}/runs", query=query
        )

    async def aget_annotation_queue_run(self, queue_id: str, index: int) -> Any:
        """Async fetch a queue run by its review index."""

        return await self.arequest_json(
            "GET", f"/api/v1/annotation-queues/{queue_id}/run/{index}"
        )

    async def aremove_run_from_annotation_queue(self, queue_id: str, run_id: str) -> Any:
        """Async remove one run from an annotation queue."""

        return await self.arequest_json(
            "DELETE", f"/api/v1/annotation-queues/{queue_id}/runs/{run_id}"
        )

    async def acreate_tool(self, *, body: Mapping[str, Any]) -> Any:
        """Async create for a workspace platform tool."""

        return await self.arequest_json("POST", "/api/v1/platform/tools", body=dict(body))

    async def alist_tools(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async list for workspace platform tools."""

        return await self.arequest_json("GET", "/api/v1/platform/tools", query=query)

    async def aget_tool_by_id(self, tool_id: str) -> Any:
        """Async fetch for a workspace tool by UUID."""

        return await self.arequest_json("GET", f"/api/v1/platform/tools/id/{tool_id}")

    async def aget_tool_by_handle(self, handle: str) -> Any:
        """Async fetch for a workspace tool by handle."""

        return await self.arequest_json("GET", f"/api/v1/platform/tools/{handle}")

    async def aupdate_tool_by_id(self, tool_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async patch for a workspace tool by UUID."""

        return await self.arequest_json(
            "PATCH",
            f"/api/v1/platform/tools/id/{tool_id}",
            body=dict(body),
        )

    async def aupdate_tool_by_handle(self, handle: str, *, body: Mapping[str, Any]) -> Any:
        """Async patch for a workspace tool by handle."""

        return await self.arequest_json(
            "PATCH",
            f"/api/v1/platform/tools/{handle}",
            body=dict(body),
        )

    async def adelete_tool_by_id(self, tool_id: str) -> Any:
        """Async deletion for a workspace tool by UUID."""

        return await self.arequest_json("DELETE", f"/api/v1/platform/tools/id/{tool_id}")

    async def adelete_tool_by_handle(self, handle: str) -> Any:
        """Async deletion for a workspace tool by handle."""

        return await self.arequest_json("DELETE", f"/api/v1/platform/tools/{handle}")

    async def alist_agents(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Async list for Fleet or Managed Deep Agents."""

        return await self.arequest_json("GET", f"{path_prefix.rstrip('/')}/agents", query=query)

    async def aiter_agents(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for Fleet or Managed Deep Agents."""

        page_query = dict(query or {})
        while True:
            page = await self.alist_agents(query=page_query, path_prefix=path_prefix)
            if not isinstance(page, Mapping):
                return
            for item in page.get("items", []) or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    async def aget_agent(
        self,
        agent_id: str,
        *,
        include_files: bool = False,
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Async fetch for an agent definition."""

        query = {"include_files": str(include_files).lower()} if include_files else None
        return await self.arequest_json(
            "GET",
            f"{path_prefix.rstrip('/')}/agents/{agent_id}",
            query=query,
        )

    async def acreate_agent(
        self,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Async create for a Fleet or Managed Deep Agent."""

        return await self.arequest_json("POST", f"{path_prefix.rstrip('/')}/agents", body=dict(body))

    async def aupdate_agent(
        self,
        agent_id: str,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Async update for a Fleet or Managed Deep Agent."""

        return await self.arequest_json(
            "PATCH",
            f"{path_prefix.rstrip('/')}/agents/{agent_id}",
            body=dict(body),
        )

    async def adelete_agent(self, agent_id: str, *, path_prefix: str = "/v1/fleet") -> Any:
        """Async deletion for a Fleet or Managed Deep Agent."""

        return await self.arequest_json("DELETE", f"{path_prefix.rstrip('/')}/agents/{agent_id}")

    async def alist_fleet_threads(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Async list for managed-agent threads."""

        return await self.arequest_json("GET", f"{path_prefix.rstrip('/')}/threads", query=query)

    async def aiter_fleet_threads(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for managed-agent threads."""

        page_query = dict(query or {})
        while True:
            page = await self.alist_fleet_threads(query=page_query, path_prefix=path_prefix)
            if not isinstance(page, Mapping):
                return
            for item in page.get("items", []) or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    async def aget_fleet_thread(self, thread_id: str, *, path_prefix: str = "/v1/fleet") -> Any:
        """Async fetch for one managed-agent thread."""

        return await self.arequest_json("GET", f"{path_prefix.rstrip('/')}/threads/{thread_id}")

    async def aupdate_fleet_thread(
        self,
        thread_id: str,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Async update for managed-agent thread metadata."""

        return await self.arequest_json(
            "PATCH",
            f"{path_prefix.rstrip('/')}/threads/{thread_id}",
            body=dict(body),
        )

    async def adelete_fleet_thread(self, thread_id: str, *, path_prefix: str = "/v1/fleet") -> Any:
        """Async deletion for a managed-agent thread."""

        return await self.arequest_json("DELETE", f"{path_prefix.rstrip('/')}/threads/{thread_id}")

    async def alist_mcp_servers(self, *, path_prefix: str = "/v1/deepagents") -> Any:
        """Async list for workspace-registered MCP servers."""

        return await self.arequest_json("GET", f"{path_prefix.rstrip('/')}/mcp-servers")

    async def alist_trigger_templates(self, *, path_prefix: str = "/v1/fleet") -> Any:
        """Async list for native Fleet trigger templates."""

        return await self.arequest_json("GET", f"{path_prefix.rstrip('/')}/trigger-templates")

    async def acreate_mcp_server(
        self,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/deepagents",
    ) -> Any:
        """Async register for an MCP server."""

        return await self.arequest_json(
            "POST",
            f"{path_prefix.rstrip('/')}/mcp-servers",
            body=dict(body),
        )

    async def aget_mcp_server(self, server_id: str, *, path_prefix: str = "/v1/deepagents") -> Any:
        """Async fetch for a registered MCP server."""

        return await self.arequest_json("GET", f"{path_prefix.rstrip('/')}/mcp-servers/{server_id}")

    async def aupdate_mcp_server(
        self,
        server_id: str,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/deepagents",
    ) -> Any:
        """Async update for a registered MCP server."""

        return await self.arequest_json(
            "PATCH",
            f"{path_prefix.rstrip('/')}/mcp-servers/{server_id}",
            body=dict(body),
        )

    async def adelete_mcp_server(self, server_id: str, *, path_prefix: str = "/v1/deepagents") -> Any:
        """Async deletion for a registered MCP server."""

        return await self.arequest_json("DELETE", f"{path_prefix.rstrip('/')}/mcp-servers/{server_id}")

    async def astore_put(self, *, namespace: list[str], key: str, value: Any) -> Any:
        """Async write for a long-term memory item."""

        return await self.arequest_json(
            "PUT",
            "/store/items",
            body={"namespace": namespace, "key": key, "value": value},
        )

    async def astore_get(self, *, namespace: list[str], key: str) -> Any:
        """Async retrieve for a long-term memory item."""

        return await self.arequest_json(
            "GET",
            "/store/items",
            query={"namespace": namespace, "key": key},
        )

    async def astore_search(
        self,
        *,
        namespace_prefix: list[str],
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async search for long-term memory items."""

        body = {"namespace_prefix": namespace_prefix, **dict(query or {})}
        return await self.arequest_json("POST", "/store/items/search", body=body)

    async def astore_delete(self, *, namespace: list[str], key: str) -> Any:
        """Async deletion for a long-term memory item."""

        return await self.arequest_json(
            "DELETE",
            "/store/items",
            query={"namespace": namespace, "key": key},
        )

    async def acreate_run_wait(self, *, body: Mapping[str, Any]) -> Any:
        """Async stateless run that waits for final output."""

        return await self.arequest_json("POST", "/runs/wait", body=dict(body))

    async def acreate_background_run(self, *, body: Mapping[str, Any]) -> Any:
        """Async stateless run that returns without waiting."""

        return await self.arequest_json("POST", "/runs", body=dict(body))

    async def asearch_runs(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async search for deployment runs."""

        return await self.arequest_json("POST", "/runs/search", body=dict(body or {}))

    async def ahealth_check(self, *, check_db: bool = False) -> Any:
        """Async Agent Server health check."""

        return await self.arequest_json("GET", "/ok", query={"check_db": int(check_db)})

    async def aserver_info(self) -> Any:
        """Async fetch for Agent Server version and feature metadata."""

        return await self.arequest_json("GET", "/info")

    async def aapi_documentation(self) -> Any:
        """Async fetch for Agent Server API documentation."""

        return await self.arequest_json("GET", "/docs")

    async def asystem_metrics(self, *, format: str = "prometheus") -> Any:
        """Async fetch for Agent Server metrics."""

        return await self.arequest_json("GET", "/metrics", query={"format": format})

    async def acancel_run(self, thread_id: str, run_id: str) -> Any:
        """Async request to cancel one active deployment run."""

        return await self.arequest_json("POST", f"/threads/{thread_id}/runs/{run_id}/cancel")

    async def acreate_feedback(
        self,
        *,
        run_id: str,
        key: str,
        score: float | None = None,
        value: Any | None = None,
        comment: str | None = None,
        trace_id: str | None = None,
        correction: Any | None = None,
        source_info: Mapping[str, Any] | None = None,
        feedback_id: str | None = None,
        source_run_id: str | None = None,
        feedback_group_id: str | None = None,
        comparative_experiment_id: str | None = None,
        session_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
        error: bool | None = None,
        feedback_source_type: str | None = None,
        extend_trace_retention: bool | None = None,
    ) -> Any:
        """Async attach feedback while preserving native LangSmith fields."""

        body: dict[str, Any] = {"run_id": run_id, "key": key}
        optional = {
            "score": score,
            "value": value,
            "comment": comment,
            "trace_id": trace_id,
            "correction": correction,
            "source_info": dict(source_info) if source_info is not None else None,
            "id": feedback_id,
            "source_run_id": source_run_id,
            "feedback_group_id": feedback_group_id,
            "comparative_experiment_id": comparative_experiment_id,
            "session_id": session_id,
            "extra": dict(extra) if extra is not None else None,
            "error": error,
            "feedback_source_type": feedback_source_type,
            "extend_trace_retention": extend_trace_retention,
        }
        body.update({name: item for name, item in optional.items() if item is not None})
        return await self.arequest_json("POST", "/api/v1/feedback", body=body)

    async def aa2a_json_rpc(
        self,
        assistant_id: str,
        *,
        body: Mapping[str, Any],
        accept: str = "application/json",
    ) -> Any:
        """Async JSON-RPC request to a LangSmith Agent-to-Agent endpoint."""

        return await self.arequest_json(
            "POST",
            f"/a2a/{assistant_id}",
            body=dict(body),
            headers={"Accept": accept},
        )

    async def aa2a_stream(
        self,
        assistant_id: str,
        *,
        body: Mapping[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Async JSON-RPC event stream from a LangSmith Agent-to-Agent endpoint."""

        async for event in self.astream_events("POST", f"/a2a/{assistant_id}", body=dict(body)):
            yield event

    async def amcp_get(self) -> Any:
        """Async open for the stateless streamable-HTTP MCP endpoint."""

        return await self.arequest_json("GET", "/mcp/")

    async def amcp_post(
        self,
        *,
        body: Mapping[str, Any],
        accept: str = "application/json",
    ) -> Any:
        """Async JSON-RPC request to the stateless MCP endpoint."""

        return await self.arequest_json(
            "POST",
            "/mcp/",
            body=dict(body),
            headers={"Accept": accept},
        )

    async def amcp_terminate(self) -> Any:
        """Async termination for an MCP session."""

        return await self.arequest_json("DELETE", "/mcp/")

    async def acreate_cron(self, *, body: Mapping[str, Any]) -> Any:
        """Async schedule for stateless runs on a new thread per execution."""

        return await self.arequest_json("POST", "/runs/crons", body=dict(body))

    async def acreate_thread_cron(self, thread_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async schedule for runs against an existing thread."""

        return await self.arequest_json(
            "POST",
            f"/threads/{thread_id}/runs/crons",
            body=dict(body),
        )

    async def asearch_crons(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async search for scheduled cron jobs."""

        return await self.arequest_json("POST", "/runs/crons/search", body=dict(body or {}))

    async def acount_crons(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Async count for scheduled cron jobs."""

        return await self.arequest_json("POST", "/runs/crons/count", body=dict(body or {}))

    async def aget_cron(self, cron_id: str) -> Any:
        """Async fetch for one scheduled cron job."""

        return await self.arequest_json("GET", f"/runs/crons/{cron_id}")

    async def aupdate_cron(self, cron_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async update for one scheduled cron job."""

        return await self.arequest_json("PATCH", f"/runs/crons/{cron_id}", body=dict(body))

    async def adelete_cron(self, cron_id: str) -> Any:
        """Async deletion for one scheduled cron job."""

        return await self.arequest_json("DELETE", f"/runs/crons/{cron_id}")

    def get_thread(self, thread_id: str) -> Any:
        """Fetch a LangSmith/LangGraph deployment thread."""

        return self.request_json("GET", f"/threads/{thread_id}")

    def patch_thread(self, thread_id: str, *, body: Mapping[str, Any]) -> Any:
        """Patch thread metadata using native Agent Server semantics."""

        return self.request_json("PATCH", f"/threads/{thread_id}", body=dict(body))

    def copy_thread(self, thread_id: str) -> Any:
        """Copy a thread and its persisted state."""

        return self.request_json("POST", f"/threads/{thread_id}/copy")

    def search_threads(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Search deployment threads using the native request body."""

        return self.request_json("POST", "/threads/search", body=dict(body or {}))

    def count_threads(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Count deployment threads matching the native criteria."""

        return self.request_json("POST", "/threads/count", body=dict(body or {}))

    def prune_threads(self, *, thread_ids: list[str], strategy: str) -> Any:
        """Prune thread checkpoints or delete threads using the native strategy."""

        return self.request_json(
            "POST",
            "/threads/prune",
            body={"thread_ids": thread_ids, "strategy": strategy},
        )

    def delete_thread(self, thread_id: str) -> Any:
        """Delete a deployment thread."""

        return self.request_json("DELETE", f"/threads/{thread_id}")

    def thread_history(self, thread_id: str, *, body: Mapping[str, Any] | None = None) -> Any:
        """Read a deployment thread history."""

        return self.request_json("POST", f"/threads/{thread_id}/history", body=dict(body or {}))

    def get_thread_history(self, thread_id: str, *, limit: int | None = None) -> Any:
        """Read thread history through the native GET convenience endpoint."""

        return self.request_json("GET", f"/threads/{thread_id}/history", query={"limit": limit})

    def join_thread_stream(
        self,
        thread_id: str,
        *,
        stream_modes: str | list[str] | None = None,
        last_event_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Subscribe to the native SSE stream for all runs on a thread."""

        query: dict[str, Any] = {}
        if stream_modes is not None:
            query["stream_modes"] = stream_modes
        headers = {"Last-Event-ID": last_event_id} if last_event_id is not None else None
        return self.stream_events(
            "GET",
            f"/threads/{thread_id}/stream",
            query=query,
            headers=headers,
        )

    def resolve_interrupt(self, thread_id: str) -> Any:
        """Resolve a paused human-in-the-loop thread."""

        return self.request_json("POST", f"/threads/{thread_id}/resolve-interrupt")

    def create_thread(self, *, metadata: Mapping[str, Any] | None = None) -> Any:
        """Create a thread for a deployed graph or Agent Server runtime."""

        return self.request_json("POST", "/threads", body={"metadata": dict(metadata or {})})

    def health_check(self, *, check_db: bool = False) -> Any:
        """Check Agent Server health, optionally including database connectivity."""

        return self.request_json("GET", "/ok", query={"check_db": int(check_db)})

    def server_info(self) -> Any:
        """Fetch Agent Server version, feature flags, and deployment metadata."""

        return self.request_json("GET", "/info")

    def api_documentation(self) -> Any:
        """Fetch the Agent Server's local HTML API documentation."""

        return self.request_json("GET", "/docs")

    def system_metrics(self, *, format: str = "prometheus") -> Any:
        """Fetch Agent Server metrics in Prometheus or JSON format."""

        return self.request_json("GET", "/metrics", query={"format": format})

    def stream_thread_run(
        self,
        thread_id: str,
        *,
        assistant_id: str,
        input: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream a deployed LangGraph run through the native SSE endpoint."""

        return self.stream_events(
            "POST",
            f"/threads/{thread_id}/runs/stream",
            body={"assistant_id": assistant_id, "input": input},
        )

    def create_assistant(self, *, graph_id: str, config: Mapping[str, Any] | None = None) -> Any:
        """Create an assistant configuration for a deployed LangGraph."""

        return self.request_json("POST", "/assistants", body={"graph_id": graph_id, "config": dict(config or {})})

    def update_assistant(self, assistant_id: str, *, config: Mapping[str, Any]) -> Any:
        """Create a new assistant version using the native full-config update shape."""

        return self.request_json("PATCH", f"/assistants/{assistant_id}", body={"config": dict(config)})

    def get_assistant(self, assistant_id: str) -> Any:
        """Fetch an assistant and its active configuration."""

        return self.request_json("GET", f"/assistants/{assistant_id}")

    def delete_assistant(self, assistant_id: str) -> Any:
        """Delete an assistant and its versions."""

        return self.request_json("DELETE", f"/assistants/{assistant_id}")

    def search_assistants(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Search assistants using the native request body."""

        return self.request_json("POST", "/assistants/search", body=dict(body or {}))

    def count_assistants(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Count assistants matching the native search criteria."""

        return self.request_json("POST", "/assistants/count", body=dict(body or {}))

    def get_assistant_graph(self, assistant_id: str, *, xray: bool | int | None = None) -> Any:
        """Fetch an assistant graph definition for inspection or visualization."""

        return self.request_json(
            "GET",
            f"/assistants/{assistant_id}/graph",
            query={"xray": xray} if xray is not None else None,
        )

    def get_assistant_schemas(self, assistant_id: str) -> Any:
        """Fetch the input, output, and config schemas for an assistant."""

        return self.request_json("GET", f"/assistants/{assistant_id}/schemas")

    def get_assistant_subgraphs(
        self,
        assistant_id: str,
        *,
        namespace: str | None = None,
    ) -> Any:
        """Fetch all assistant subgraphs or those under one namespace."""

        path = f"/assistants/{assistant_id}/subgraphs"
        if namespace is not None:
            path = f"{path}/{quote(namespace, safe='')}"
        return self.request_json("GET", path)

    def get_assistant_versions(self, assistant_id: str, *, query: Mapping[str, Any] | None = None) -> Any:
        """List all versions of an assistant."""

        return self.request_json("GET", f"/assistants/{assistant_id}/versions", query=query)

    def set_latest_assistant_version(self, assistant_id: str, version: int) -> Any:
        """Select the active version for an assistant."""

        return self.request_json(
            "POST",
            f"/assistants/{assistant_id}/latest",
            body={"version": version},
        )

    def create_thread_run(
        self,
        thread_id: str,
        *,
        assistant_id: str,
        input: Any,
        stream: bool = False,
        command: Mapping[str, Any] | None = None,
    ) -> Any:
        """Start a non-streaming or streaming thread run."""

        body = {"assistant_id": assistant_id, "input": input}
        if command is not None:
            body["command"] = dict(command)
        if stream:
            return self.stream_events("POST", f"/threads/{thread_id}/runs/stream", body=body)
        return self.request_json("POST", f"/threads/{thread_id}/runs", body=body)

    def create_background_run(self, *, body: Mapping[str, Any]) -> Any:
        """Start a stateless run and return without waiting for its output."""

        return self.request_json("POST", "/runs", body=dict(body))

    def create_run_wait(self, *, body: Mapping[str, Any]) -> Any:
        """Start a stateless run and wait for its final output."""

        return self.request_json("POST", "/runs/wait", body=dict(body))

    def create_run_stream(self, *, body: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Start a stateless run and stream its native SSE output."""

        return self.stream_events("POST", "/runs/stream", body=dict(body))

    def create_run_batch(self, *, body: Mapping[str, Any]) -> Any:
        """Submit a native batch of stateless runs."""

        return self.request_json("POST", "/runs/batch", body=dict(body))

    def a2a_json_rpc(
        self,
        assistant_id: str,
        *,
        body: Mapping[str, Any],
        accept: str = "application/json",
    ) -> Any:
        """Send a JSON-RPC request to a LangSmith Agent-to-Agent endpoint."""

        return self.request_json(
            "POST",
            f"/a2a/{assistant_id}",
            body=dict(body),
            headers={"Accept": accept},
        )

    def a2a_stream(self, assistant_id: str, *, body: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream JSON-RPC events from a LangSmith Agent-to-Agent endpoint."""

        return self.stream_events("POST", f"/a2a/{assistant_id}", body=dict(body))

    def mcp_get(self) -> Any:
        """Open the stateless streamable-HTTP MCP endpoint."""

        return self.request_json("GET", "/mcp/")

    def mcp_post(
        self,
        *,
        body: Mapping[str, Any],
        accept: str = "application/json",
    ) -> Any:
        """Send a JSON-RPC request to the stateless MCP endpoint."""

        return self.request_json("POST", "/mcp/", body=dict(body), headers={"Accept": accept})

    def mcp_terminate(self) -> Any:
        """Terminate an MCP session when the deployment exposes session state."""

        return self.request_json("DELETE", "/mcp/")

    def create_cron(self, *, body: Mapping[str, Any]) -> Any:
        """Schedule stateless runs on a new thread for each execution."""

        return self.request_json("POST", "/runs/crons", body=dict(body))

    def create_thread_cron(self, thread_id: str, *, body: Mapping[str, Any]) -> Any:
        """Schedule runs against an existing thread."""

        return self.request_json("POST", f"/threads/{thread_id}/runs/crons", body=dict(body))

    def search_crons(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Search scheduled cron jobs using native filter fields."""

        return self.request_json("POST", "/runs/crons/search", body=dict(body or {}))

    def count_crons(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Count scheduled cron jobs matching native filters."""

        return self.request_json("POST", "/runs/crons/count", body=dict(body or {}))

    def get_cron(self, cron_id: str) -> Any:
        """Fetch one scheduled cron job."""

        return self.request_json("GET", f"/runs/crons/{cron_id}")

    def update_cron(self, cron_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update a scheduled cron job."""

        return self.request_json("PATCH", f"/runs/crons/{cron_id}", body=dict(body))

    def delete_cron(self, cron_id: str) -> Any:
        """Delete a scheduled cron job."""

        return self.request_json("DELETE", f"/runs/crons/{cron_id}")

    def get_thread_state(self, thread_id: str, *, checkpoint_id: str | None = None) -> Any:
        """Read the current remote graph state, optionally at a checkpoint."""

        return self.request_json(
            "GET",
            f"/threads/{thread_id}/state",
            query={"checkpoint_id": checkpoint_id},
        )

    def get_thread_state_at_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        subgraphs: bool | None = None,
    ) -> Any:
        """Read a thread's state at a specific checkpoint."""

        return self.request_json(
            "GET",
            f"/threads/{thread_id}/state/{checkpoint_id}",
            query={"subgraphs": subgraphs},
        )

    def get_thread_state_at_checkpoint_body(
        self,
        thread_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Read checkpoint state using the native POST checkpoint payload."""

        return self.request_json("POST", f"/threads/{thread_id}/state/checkpoint", body=dict(body))

    def update_thread_state(self, thread_id: str, *, values: Any, as_node: str | None = None) -> Any:
        """Update remote graph state using native state-update fields."""

        body: dict[str, Any] = {"values": values}
        if as_node is not None:
            body["as_node"] = as_node
        return self.request_json("POST", f"/threads/{thread_id}/state", body=body)

    def get_run(self, thread_id: str, run_id: str) -> Any:
        """Fetch one run from a deployment thread."""

        return self.request_json("GET", f"/threads/{thread_id}/runs/{run_id}")

    def list_thread_runs(self, thread_id: str, *, query: Mapping[str, Any] | None = None) -> Any:
        """List runs belonging to a thread."""

        return self.request_json("GET", f"/threads/{thread_id}/runs", query=query)

    def list_run_events(self, thread_id: str, run_id: str) -> Any:
        """Fetch persisted events for a thread run."""

        return self.request_json("GET", f"/threads/{thread_id}/runs/{run_id}/events")

    def join_run(self, thread_id: str, run_id: str) -> Any:
        """Wait for a thread run to finish and return its final payload."""

        return self.request_json("GET", f"/threads/{thread_id}/runs/{run_id}/join")

    def join_run_stream(self, thread_id: str, run_id: str) -> Iterator[dict[str, Any]]:
        """Stream a persisted thread run to completion."""

        return self.stream_events("GET", f"/threads/{thread_id}/runs/{run_id}/join")

    def cancel_runs(self, thread_id: str, *, body: Mapping[str, Any] | None = None) -> Any:
        """Cancel multiple active runs in a thread."""

        return self.request_json("POST", f"/threads/{thread_id}/runs/cancel", body=dict(body or {}))

    def cancel_run(self, thread_id: str, run_id: str) -> Any:
        """Request cancellation of an active deployment run."""

        return self.request_json("POST", f"/threads/{thread_id}/runs/{run_id}/cancel")

    def delete_run(self, thread_id: str, run_id: str) -> Any:
        """Delete a deployment run."""

        return self.request_json("DELETE", f"/threads/{thread_id}/runs/{run_id}")

    def create_dataset(
        self,
        *,
        name: str,
        description: str | None = None,
        data_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Create a LangSmith dataset without requiring the LangSmith SDK."""

        body: dict[str, Any] = {"name": name}
        optional = {
            "description": description,
            "data_type": data_type,
            "metadata": dict(metadata) if metadata is not None else None,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self.request_json("POST", "/api/v1/datasets", body=body)

    def list_datasets(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List datasets with native filter, pagination, and sorting fields."""

        return self.request_json("GET", "/api/v1/datasets", query=query)

    def iter_datasets(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate cursor-paginated datasets without dropping caller filters."""

        page_query = dict(query or {})
        while True:
            page = self.list_datasets(query=page_query)
            if not isinstance(page, Mapping):
                return
            items = page.get("datasets", page.get("items", [])) or []
            for item in items:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor") or page.get("cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    def get_dataset(self, dataset_id: str) -> Any:
        """Fetch one dataset by ID."""

        return self.request_json("GET", f"/api/v1/datasets/{dataset_id}")

    def update_dataset(self, dataset_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update dataset metadata using native patch fields."""

        return self.request_json("PATCH", f"/api/v1/datasets/{dataset_id}", body=dict(body))

    def delete_dataset(self, dataset_id: str) -> Any:
        """Delete one dataset and its examples."""

        return self.request_json("DELETE", f"/api/v1/datasets/{dataset_id}")

    def read_dataset_version(
        self,
        dataset_id: str,
        *,
        as_of: str | None = None,
        tag: str | None = None,
    ) -> Any:
        """Fetch a dataset version by timestamp or tag."""

        if (as_of is None) == (tag is None):
            raise ValueError("Exactly one of as_of and tag must be specified.")
        query = {"as_of": as_of} if as_of is not None else {"tag": tag}
        return self.request_json("GET", f"/api/v1/datasets/{dataset_id}/version", query=query)

    def list_dataset_versions(
        self,
        dataset_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """List version history for a dataset."""

        return self.request_json("GET", f"/api/v1/datasets/{dataset_id}/versions", query=query)

    def diff_dataset_versions(
        self,
        dataset_id: str,
        *,
        from_version: str,
        to_version: str,
    ) -> Any:
        """Return additions/removals between two dataset versions."""

        return self.request_json(
            "GET",
            f"/api/v1/datasets/{dataset_id}/versions/diff",
            query={"from_version": from_version, "to_version": to_version},
        )

    def list_dataset_splits(self, dataset_id: str, *, as_of: str | None = None) -> Any:
        """List split names for a dataset version."""

        query = {"as_of": as_of} if as_of is not None else None
        return self.request_json("GET", f"/api/v1/datasets/{dataset_id}/splits", query=query)

    def update_dataset_splits(
        self,
        dataset_id: str,
        *,
        split_name: str,
        example_ids: list[str],
        remove: bool = False,
    ) -> Any:
        """Add or remove examples from a dataset split."""

        return self.request_json(
            "PUT",
            f"/api/v1/datasets/{dataset_id}/splits",
            body={
                "split_name": split_name,
                "examples": list(example_ids),
                "remove": remove,
            },
        )

    def share_dataset(self, dataset_id: str) -> Any:
        """Create or refresh a public share for a dataset."""

        return self.request_json(
            "PUT",
            f"/api/v1/datasets/{dataset_id}/share",
            body={"dataset_id": dataset_id},
        )

    def unshare_dataset(self, dataset_id: str) -> Any:
        """Remove the public share for a dataset."""

        return self.request_json("DELETE", f"/api/v1/datasets/{dataset_id}/share")

    def read_dataset_delta(
        self,
        dataset_id: str,
        *,
        baseline_session_id: str,
        comparison_session_ids: list[str],
        feedback_key: str,
        filters: Mapping[str, Any] | None = None,
        offset: int = 0,
        limit: int = 100,
        comparative_experiment_id: str | None = None,
    ) -> Any:
        """Compare feedback regressions and improvements across sessions."""

        body: dict[str, Any] = {
            "baseline_session_id": baseline_session_id,
            "comparison_session_ids": list(comparison_session_ids),
            "feedback_key": feedback_key,
            "offset": offset,
            "limit": limit,
        }
        if filters is not None:
            body["filters"] = dict(filters)
        if comparative_experiment_id is not None:
            body["comparative_experiment_id"] = comparative_experiment_id
        return self.request_json(
            "POST", f"/api/v1/datasets/{dataset_id}/runs/delta", body=body
        )

    def read_shared_dataset_examples_with_runs(
        self,
        share_token: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Read examples and associated runs from a shared dataset."""

        return self.request_json(
            "POST",
            f"/api/v1/public/{share_token}/examples/runs",
            body=dict(body),
        )

    def create_example(
        self,
        *,
        dataset_id: str,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        example_id: str | None = None,
        name: str | None = None,
    ) -> Any:
        """Create one dataset example with optional reference output and metadata."""

        body: dict[str, Any] = {"dataset_id": dataset_id, "inputs": dict(inputs)}
        optional = {
            "outputs": dict(outputs) if outputs is not None else None,
            "metadata": dict(metadata) if metadata is not None else None,
            "id": example_id,
            "name": name,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self.request_json("POST", "/api/v1/examples", body=body)

    def list_examples(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List dataset examples with native query fields."""

        return self.request_json("GET", "/api/v1/examples", query=query)

    def iter_examples(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate cursor-paginated examples without dropping dataset filters."""

        page_query = dict(query or {})
        while True:
            page = self.list_examples(query=page_query)
            if not isinstance(page, Mapping):
                return
            items = page.get("examples", page.get("items", [])) or []
            for item in items:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor") or page.get("cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    def get_example(self, example_id: str) -> Any:
        """Fetch one dataset example by ID."""

        return self.request_json("GET", f"/api/v1/examples/{example_id}")

    def update_example(self, example_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update an example's inputs, outputs, or metadata."""

        return self.request_json("PATCH", f"/api/v1/examples/{example_id}", body=dict(body))

    def delete_example(self, example_id: str) -> Any:
        """Delete one dataset example."""

        return self.request_json("DELETE", f"/api/v1/examples/{example_id}")

    def delete_examples(self, example_ids: list[str], *, hard_delete: bool = False) -> Any:
        """Delete multiple examples, optionally using the hard-delete endpoint."""

        if hard_delete:
            return self.request_json(
                "POST",
                "/api/v1/platform/datasets/examples/delete",
                body={"example_ids": list(example_ids), "hard_delete": True},
            )
        return self.request_json(
            "DELETE", "/api/v1/examples", query={"example_ids": list(example_ids)}
        )

    def list_shared_examples(
        self,
        share_token: str,
        *,
        example_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> Any:
        """List examples from a public dataset share."""

        query: dict[str, Any] = {}
        if example_ids is not None:
            query["id"] = list(example_ids)
        if limit is not None:
            query["limit"] = limit
        return self.request_json(
            "GET", f"/api/v1/public/{share_token}/examples", query=query or None
        )

    def read_dataset_openai_finetuning(self, dataset_id: str) -> Any:
        """Download a dataset in OpenAI fine-tuning JSONL format."""

        return self.request_json("GET", f"/api/v1/datasets/{dataset_id}/openai_ft")

    def search_runs(self, *, body: Mapping[str, Any] | None = None) -> Any:
        """Search deployment runs using the native request body."""

        return self.request_json("POST", "/runs/search", body=dict(body or {}))

    def create_feedback(
        self,
        *,
        run_id: str,
        key: str,
        score: float | None = None,
        value: Any | None = None,
        comment: str | None = None,
        correction: Any | None = None,
        source_info: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        feedback_id: str | None = None,
        source_run_id: str | None = None,
        feedback_group_id: str | None = None,
        comparative_experiment_id: str | None = None,
        session_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
        error: bool | None = None,
        feedback_source_type: str | None = None,
        extend_trace_retention: bool | None = None,
    ) -> Any:
        """Attach feedback while preserving native LangSmith fields."""

        body: dict[str, Any] = {"run_id": run_id, "key": key}
        optional = {
            "score": score,
            "value": value,
            "comment": comment,
            "correction": correction,
            "source_info": dict(source_info) if source_info is not None else None,
            "trace_id": trace_id,
            "id": feedback_id,
            "source_run_id": source_run_id,
            "feedback_group_id": feedback_group_id,
            "comparative_experiment_id": comparative_experiment_id,
            "session_id": session_id,
            "extra": dict(extra) if extra is not None else None,
            "error": error,
            "feedback_source_type": feedback_source_type,
            "extend_trace_retention": extend_trace_retention,
        }
        body.update({name: item for name, item in optional.items() if item is not None})
        return self.request_json("POST", "/api/v1/feedback", body=body)

    def list_feedback(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List feedback records using the native filter query."""

        return self.request_json("GET", "/api/v1/feedback", query=query)

    def get_feedback(self, feedback_id: str) -> Any:
        """Fetch one feedback record."""

        return self.request_json("GET", f"/api/v1/feedback/{feedback_id}")

    def update_feedback(self, feedback_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update a feedback record with native patch fields."""

        return self.request_json(
            "PATCH",
            f"/api/v1/feedback/{feedback_id}",
            body=dict(body),
        )

    def delete_feedback(self, feedback_id: str) -> Any:
        """Delete one feedback record."""

        return self.request_json("DELETE", f"/api/v1/feedback/{feedback_id}")

    def create_feedback_config(
        self,
        feedback_key: str,
        *,
        feedback_config: Mapping[str, Any],
        is_lower_score_better: bool = False,
    ) -> Any:
        """Create a LangSmith feedback configuration."""

        return self.request_json(
            "POST",
            "/api/v1/feedback-configs",
            body={
                "feedback_key": feedback_key,
                "feedback_config": dict(feedback_config),
                "is_lower_score_better": is_lower_score_better,
            },
        )

    def list_feedback_configs(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List LangSmith feedback configurations."""

        return self.request_json("GET", "/api/v1/feedback-configs", query=query)

    def update_feedback_config(self, feedback_key: str, *, body: Mapping[str, Any]) -> Any:
        """Update a LangSmith feedback configuration."""

        payload = {"feedback_key": feedback_key, **dict(body)}
        return self.request_json("PATCH", "/api/v1/feedback-configs", body=payload)

    def delete_feedback_config(self, feedback_key: str) -> Any:
        """Soft-delete a LangSmith feedback configuration."""

        return self.request_json(
            "DELETE", "/api/v1/feedback-configs", query={"feedback_key": feedback_key}
        )

    def create_presigned_feedback_token(
        self,
        run_id: str,
        feedback_key: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        """Create a browser-safe presigned feedback ingest token."""

        payload = {"run_id": run_id, "feedback_key": feedback_key, **dict(body or {})}
        return self.request_json("POST", "/api/v1/feedback/tokens", body=payload)

    def list_presigned_feedback_tokens(
        self, run_id: str, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """List presigned feedback tokens for a run."""

        payload = {"run_id": run_id, **dict(query or {})}
        return self.request_json("GET", "/api/v1/feedback/tokens", query=payload)

    def create_agent_connection(
        self,
        agent_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Create an Agent Auth connection for a LangSmith agent."""

        return self.request_json(
            "POST",
            f"/v2/auth/agents/{agent_id}/connections",
            body=dict(body),
        )

    def list_agent_connections(self, agent_id: str) -> Any:
        """List Agent Auth connections configured for a LangSmith agent."""

        return self.request_json("GET", f"/v2/auth/agents/{agent_id}/connections")

    def remove_agent_connection(self, agent_id: str, connection_id: str) -> Any:
        """Remove one Agent Auth connection from a LangSmith agent."""

        return self.request_json(
            "DELETE",
            f"/v2/auth/agents/{agent_id}/connections/{connection_id}",
        )

    def list_connection_tokens(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List Fleet Agent Auth connection tokens."""

        return self.request_json("GET", "/v1/fleet/auth-tokens", query=query)

    def update_connection_token(self, token_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update connection-token metadata such as label/default."""

        return self.request_json(
            "PATCH", f"/v1/fleet/auth-tokens/{token_id}", body=dict(body)
        )

    def revoke_connection_token(self, token_id: str) -> Any:
        """Revoke one Fleet Agent Auth connection token."""

        return self.request_json("DELETE", f"/v1/fleet/auth-tokens/{token_id}")

    def create_annotation_queue(
        self,
        *,
        name: str,
        description: str | None = None,
        queue_id: str | None = None,
        rubric_instructions: str | None = None,
        rubric_items: list[Mapping[str, Any]] | None = None,
    ) -> Any:
        """Create a LangSmith human-review annotation queue."""

        body: dict[str, Any] = {"name": name}
        optional = {
            "description": description,
            "id": queue_id,
            "rubric_instructions": rubric_instructions,
            "rubric_items": rubric_items,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self.request_json("POST", "/api/v1/annotation-queues", body=body)

    def list_annotation_queues(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List LangSmith annotation queues."""

        return self.request_json("GET", "/api/v1/annotation-queues", query=query)

    def get_annotation_queue(self, queue_id: str) -> Any:
        """Fetch one annotation queue."""

        return self.request_json("GET", f"/api/v1/annotation-queues/{queue_id}")

    def update_annotation_queue(self, queue_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update annotation queue metadata and rubric."""

        return self.request_json(
            "PATCH", f"/api/v1/annotation-queues/{queue_id}", body=dict(body)
        )

    def delete_annotation_queue(self, queue_id: str) -> Any:
        """Delete one annotation queue."""

        return self.request_json("DELETE", f"/api/v1/annotation-queues/{queue_id}")

    def add_runs_to_annotation_queue(
        self,
        queue_id: str,
        *,
        run_ids: list[str] | None = None,
        runs: list[Mapping[str, Any]] | None = None,
    ) -> Any:
        """Add run IDs or full run keys to an annotation queue."""

        if (run_ids is None) == (runs is None):
            raise ValueError("Provide exactly one of run_ids or runs.")
        path = (
            f"/api/v1/annotation-queues/{queue_id}/runs"
            if run_ids is not None
            else f"/api/v1/annotation-queues/{queue_id}/runs/by-key"
        )
        payload: Any = list(run_ids) if run_ids is not None else [dict(run) for run in runs or []]
        return self.request_json("POST", path, body=payload)

    def list_annotation_queue_runs(
        self, queue_id: str, *, query: Mapping[str, Any] | None = None
    ) -> Any:
        """List runs assigned to an annotation queue."""

        return self.request_json(
            "GET", f"/api/v1/annotation-queues/{queue_id}/runs", query=query
        )

    def get_annotation_queue_run(self, queue_id: str, index: int) -> Any:
        """Fetch a queue run by its review index."""

        return self.request_json("GET", f"/api/v1/annotation-queues/{queue_id}/run/{index}")

    def remove_run_from_annotation_queue(self, queue_id: str, run_id: str) -> Any:
        """Remove one run from an annotation queue."""

        return self.request_json(
            "DELETE", f"/api/v1/annotation-queues/{queue_id}/runs/{run_id}"
        )

    def create_tool(self, *, body: Mapping[str, Any]) -> Any:
        """Create a workspace tool in the LangSmith platform registry."""

        return self.request_json("POST", "/api/v1/platform/tools", body=dict(body))

    def list_tools(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List workspace tools with native filters and pagination."""

        return self.request_json("GET", "/api/v1/platform/tools", query=query)

    def get_tool_by_id(self, tool_id: str) -> Any:
        """Fetch a workspace tool by UUID."""

        return self.request_json("GET", f"/api/v1/platform/tools/id/{tool_id}")

    def get_tool_by_handle(self, handle: str) -> Any:
        """Fetch a workspace tool by stable handle."""

        return self.request_json("GET", f"/api/v1/platform/tools/{handle}")

    def update_tool_by_id(self, tool_id: str, *, body: Mapping[str, Any]) -> Any:
        """Patch a workspace tool by UUID."""

        return self.request_json("PATCH", f"/api/v1/platform/tools/id/{tool_id}", body=dict(body))

    def update_tool_by_handle(self, handle: str, *, body: Mapping[str, Any]) -> Any:
        """Patch a workspace tool by stable handle."""

        return self.request_json("PATCH", f"/api/v1/platform/tools/{handle}", body=dict(body))

    def delete_tool_by_id(self, tool_id: str) -> Any:
        """Delete a workspace tool by UUID."""

        return self.request_json("DELETE", f"/api/v1/platform/tools/id/{tool_id}")

    def delete_tool_by_handle(self, handle: str) -> Any:
        """Delete a workspace tool by stable handle."""

        return self.request_json("DELETE", f"/api/v1/platform/tools/{handle}")

    def list_agents(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """List Fleet or Managed Deep Agents accessible to the caller."""

        return self.request_json("GET", f"{path_prefix.rstrip('/')}/agents", query=query)

    def iter_agents(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> Iterator[dict[str, Any]]:
        """Iterate cursor-paginated Fleet or Managed Deep Agents."""

        page_query = dict(query or {})
        while True:
            page = self.list_agents(query=page_query, path_prefix=path_prefix)
            if not isinstance(page, Mapping):
                return
            for item in page.get("items", []) or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    def get_agent(
        self,
        agent_id: str,
        *,
        include_files: bool = False,
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Fetch an agent definition, optionally including its file map."""

        return self.request_json(
            "GET",
            f"{path_prefix.rstrip('/')}/agents/{agent_id}",
            query={"include_files": str(include_files).lower()} if include_files else None,
        )

    def create_agent(
        self,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Create a Fleet or Managed Deep Agent from its native definition."""

        return self.request_json("POST", f"{path_prefix.rstrip('/')}/agents", body=dict(body))

    def update_agent(
        self,
        agent_id: str,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Patch a Fleet or Managed Deep Agent using native merge semantics."""

        return self.request_json(
            "PATCH",
            f"{path_prefix.rstrip('/')}/agents/{agent_id}",
            body=dict(body),
        )

    def delete_agent(self, agent_id: str, *, path_prefix: str = "/v1/fleet") -> Any:
        """Delete a Fleet or Managed Deep Agent and its associated state."""

        return self.request_json("DELETE", f"{path_prefix.rstrip('/')}/agents/{agent_id}")

    def list_fleet_threads(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """List managed-agent threads using native pagination/filter fields."""

        return self.request_json("GET", f"{path_prefix.rstrip('/')}/threads", query=query)

    def iter_fleet_threads(
        self,
        *,
        query: Mapping[str, Any] | None = None,
        path_prefix: str = "/v1/fleet",
    ) -> Iterator[dict[str, Any]]:
        """Iterate cursor-paginated managed-agent threads."""

        page_query = dict(query or {})
        while True:
            page = self.list_fleet_threads(query=page_query, path_prefix=path_prefix)
            if not isinstance(page, Mapping):
                return
            for item in page.get("items", []) or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.get("next_cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    def get_fleet_thread(self, thread_id: str, *, path_prefix: str = "/v1/fleet") -> Any:
        """Fetch one managed-agent thread."""

        return self.request_json("GET", f"{path_prefix.rstrip('/')}/threads/{thread_id}")

    def update_fleet_thread(
        self,
        thread_id: str,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/fleet",
    ) -> Any:
        """Update managed-agent thread metadata or title."""

        return self.request_json(
            "PATCH",
            f"{path_prefix.rstrip('/')}/threads/{thread_id}",
            body=dict(body),
        )

    def delete_fleet_thread(self, thread_id: str, *, path_prefix: str = "/v1/fleet") -> Any:
        """Delete a managed-agent thread."""

        return self.request_json("DELETE", f"{path_prefix.rstrip('/')}/threads/{thread_id}")

    def list_mcp_servers(self, *, path_prefix: str = "/v1/deepagents") -> Any:
        """List workspace-registered MCP servers."""

        return self.request_json("GET", f"{path_prefix.rstrip('/')}/mcp-servers")

    def list_trigger_templates(self, *, path_prefix: str = "/v1/fleet") -> Any:
        """List native Fleet trigger templates and their configuration schemas."""

        return self.request_json("GET", f"{path_prefix.rstrip('/')}/trigger-templates")

    def create_mcp_server(
        self,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/deepagents",
    ) -> Any:
        """Register an MCP server and its native credential headers."""

        return self.request_json(
            "POST",
            f"{path_prefix.rstrip('/')}/mcp-servers",
            body=dict(body),
        )

    def get_mcp_server(self, server_id: str, *, path_prefix: str = "/v1/deepagents") -> Any:
        """Fetch a registered MCP server."""

        return self.request_json("GET", f"{path_prefix.rstrip('/')}/mcp-servers/{server_id}")

    def update_mcp_server(
        self,
        server_id: str,
        *,
        body: Mapping[str, Any],
        path_prefix: str = "/v1/deepagents",
    ) -> Any:
        """Update a registered MCP server."""

        return self.request_json(
            "PATCH",
            f"{path_prefix.rstrip('/')}/mcp-servers/{server_id}",
            body=dict(body),
        )

    def delete_mcp_server(self, server_id: str, *, path_prefix: str = "/v1/deepagents") -> Any:
        """Delete a registered MCP server."""

        return self.request_json("DELETE", f"{path_prefix.rstrip('/')}/mcp-servers/{server_id}")

    def store_put(self, *, namespace: list[str], key: str, value: Any) -> Any:
        """Store a long-term memory item through the deployment store API."""

        return self.request_json(
            "PUT",
            "/store/items",
            body={"namespace": namespace, "key": key, "value": value},
        )

    def store_get(self, *, namespace: list[str], key: str) -> Any:
        """Retrieve a long-term memory item."""

        return self.request_json(
            "GET",
            "/store/items",
            query={"namespace": namespace, "key": key},
        )

    def store_search(self, *, namespace_prefix: list[str], query: Mapping[str, Any] | None = None) -> Any:
        """Search long-term memory items under a namespace prefix."""

        body = {"namespace_prefix": namespace_prefix, **dict(query or {})}
        return self.request_json("POST", "/store/items/search", body=body)

    def store_delete(self, *, namespace: list[str], key: str) -> Any:
        """Delete a long-term memory item."""

        return self.request_json(
            "DELETE",
            "/store/items",
            query={"namespace": namespace, "key": key},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None,
        body: Any | None,
        headers: Mapping[str, str] | None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        normalized_path = path if path.startswith("/") else f"/{path}"
        query_string = urlencode(
            [(key, value) for key, value in (query or {}).items() if value is not None],
            doseq=True,
        )
        url = f"{self.base_url}{normalized_path}"
        if query_string:
            url = f"{url}?{query_string}"
        request_headers = {
            "Accept": "application/json",
            "X-Api-Key": self.api_key,
            **(headers or {}),
        }
        raw_body = None
        if body is not None:
            raw_body = json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        return self._transport(method.upper(), url, request_headers, raw_body)


class LangSmithControlPlaneClient(LangSmithAPIClient):
    """Dependency-free client for LangSmith Deployment control-plane APIs.

    Deployment control is a different service from the LangSmith data-plane API.
    The workspace identifier is therefore explicit and is sent as ``X-Tenant-Id``
    rather than being mixed into an AgentSpec or a runtime request.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tenant_id: str | None = None,
        base_url: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or os.environ.get(
                "LANGSMITH_CONTROL_PLANE_URL", "https://api.host.langchain.com"
            ),
            transport=transport,
        )
        self.tenant_id = tenant_id or os.environ.get("LANGSMITH_TENANT_ID")
        if not self.tenant_id:
            raise ValueError("LANGSMITH_TENANT_ID or tenant_id is required for control-plane calls.")

    async def alist_deployments(self, *, name_contains: str | None = None) -> Any:
        """Async list for deployments in the selected workspace."""

        query = {"name_contains": name_contains} if name_contains else None
        return await self.arequest_json("GET", "/v2/deployments", query=query)

    async def acreate_deployment(self, *, body: Mapping[str, Any]) -> Any:
        """Async create for a deployment and its initial revision."""

        return await self.arequest_json("POST", "/v2/deployments", body=dict(body))

    async def aget_deployment(self, deployment_id: str) -> Any:
        """Async fetch for deployment metadata."""

        return await self.arequest_json("GET", f"/v2/deployments/{deployment_id}")

    async def apatch_deployment(self, deployment_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async update for deployment configuration."""

        return await self.arequest_json(
            "PATCH",
            f"/v2/deployments/{deployment_id}",
            body=dict(body),
        )

    async def adelete_deployment(self, deployment_id: str) -> Any:
        """Async deletion for one deployment."""

        return await self.arequest_json("DELETE", f"/v2/deployments/{deployment_id}")

    async def adelete_deployments(self, deployment_ids: list[str]) -> Any:
        """Async bulk deletion for deployments."""

        return await self.arequest_json(
            "DELETE",
            "/v2/deployments",
            query={"deployment_ids": deployment_ids},
        )

    async def alist_revisions(self, deployment_id: str) -> Any:
        """Async list for deployment revisions."""

        return await self.arequest_json("GET", f"/v2/deployments/{deployment_id}/revisions")

    async def aget_revision(self, deployment_id: str, revision_id: str) -> Any:
        """Async fetch for one deployment revision."""

        return await self.arequest_json(
            "GET",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}",
        )

    async def acreate_deployment_revision(
        self,
        deployment_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Async create for a deployment revision."""

        return await self.arequest_json(
            "POST",
            f"/v2/deployments/{deployment_id}/revisions",
            body=dict(body),
        )

    async def aredeploy_revision(self, deployment_id: str, revision_id: str) -> Any:
        """Async redeploy for an existing revision."""

        return await self.arequest_json(
            "POST",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}/redeploy",
        )

    async def ainterrupt_deployment_revision(self, deployment_id: str, revision_id: str) -> Any:
        """Async interruption for an in-progress revision."""

        return await self.arequest_json(
            "POST",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}/interruption",
        )

    async def alist_deployment_logs(
        self,
        deployment_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list for deployment log batches."""

        return await self.arequest_json("GET", f"/v2/deployments/{deployment_id}/logs", query=query)

    async def alist_revision_logs(
        self,
        deployment_id: str,
        revision_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list for revision logs."""

        return await self.arequest_json(
            "GET",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}/logs",
            query=query,
        )

    async def alist_deployment_log_entries(
        self,
        *,
        deployment_id: str,
        revision_id: str | None = None,
        log_type: str = "DEPLOY",
        start_time: str | None = None,
        end_time: str | None = None,
        sort_order: str | None = None,
    ) -> Any:
        """Async list for individual deployment log entries."""

        query = {
            key: value
            for key, value in {
                "deployment_id": deployment_id,
                "revision_id": revision_id,
                "log_type": log_type,
                "start_time": start_time,
                "end_time": end_time,
                "sort_order": sort_order,
            }.items()
            if value is not None
        }
        return await self.arequest_json("GET", "/v2/deployment-logs", query=query)

    async def apatch_deployment_resource_tiers(
        self,
        deployment_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Async update for deployment resource tiers."""

        return await self.arequest_json(
            "PATCH",
            f"/v2/deployments/{deployment_id}/resource-tiers",
            body=dict(body),
        )

    async def apatch_deployment_tier(
        self,
        deployment_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Async update for the deployment tier."""

        return await self.arequest_json(
            "PATCH",
            f"/v2/deployments/{deployment_id}/deployment-tier",
            body=dict(body),
        )

    async def aget_free_deployment_count(self) -> Any:
        """Async fetch for the workspace free deployment count."""

        return await self.arequest_json("GET", "/v2/deployments/free-count")

    async def acreate_listener(self, *, body: Mapping[str, Any]) -> Any:
        """Async create for a LangSmith v2 listener."""

        return await self.arequest_json("POST", "/v2/listeners", body=dict(body))

    async def alist_listeners(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async list for LangSmith v2 listeners."""

        return await self.arequest_json("GET", "/v2/listeners", query=query)

    async def aget_listener(self, listener_id: str) -> Any:
        """Async fetch for one LangSmith v2 listener."""

        return await self.arequest_json("GET", f"/v2/listeners/{listener_id}")

    async def apatch_listener(self, listener_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async update for one LangSmith v2 listener."""

        return await self.arequest_json("PATCH", f"/v2/listeners/{listener_id}", body=dict(body))

    async def adelete_listener(self, listener_id: str) -> Any:
        """Async deletion for one LangSmith v2 listener."""

        return await self.arequest_json("DELETE", f"/v2/listeners/{listener_id}")

    def list_deployments(self, *, name_contains: str | None = None) -> Any:
        """List deployments in the selected LangSmith workspace."""

        query = {"name_contains": name_contains} if name_contains else None
        return self.request_json("GET", "/v2/deployments", query=query)

    def create_deployment(self, *, body: Mapping[str, Any]) -> Any:
        """Create a deployment and its initial revision."""

        return self.request_json("POST", "/v2/deployments", body=dict(body))

    def get_deployment(self, deployment_id: str) -> Any:
        """Fetch a deployment and its latest revision metadata."""

        return self.request_json("GET", f"/v2/deployments/{deployment_id}")

    def patch_deployment(self, deployment_id: str, *, body: Mapping[str, Any]) -> Any:
        """Update a deployment, optionally creating a new revision."""

        return self.request_json(
            "PATCH",
            f"/v2/deployments/{deployment_id}",
            body=dict(body),
        )

    def delete_deployment(self, deployment_id: str) -> Any:
        """Delete a deployment; the control plane normally returns 204."""

        return self.request_json("DELETE", f"/v2/deployments/{deployment_id}")

    def delete_deployments(self, deployment_ids: list[str]) -> Any:
        """Delete multiple deployments with the native partial-success contract."""

        return self.request_json("DELETE", "/v2/deployments", query={"deployment_ids": deployment_ids})

    def list_revisions(self, deployment_id: str) -> Any:
        """List revisions, normally newest first."""

        return self.request_json("GET", f"/v2/deployments/{deployment_id}/revisions")

    def get_revision(self, deployment_id: str, revision_id: str) -> Any:
        """Fetch one deployment revision and its status."""

        return self.request_json(
            "GET",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}",
        )

    def create_deployment_revision(
        self,
        deployment_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Create a new revision using the native revision payload."""

        return self.request_json(
            "POST",
            f"/v2/deployments/{deployment_id}/revisions",
            body=dict(body),
        )

    def redeploy_revision(self, deployment_id: str, revision_id: str) -> Any:
        """Redeploy an existing revision."""

        return self.request_json(
            "POST",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}/redeploy",
        )

    def interrupt_deployment_revision(self, deployment_id: str, revision_id: str) -> Any:
        """Interrupt an in-progress deployment revision."""

        return self.request_json(
            "POST",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}/interruption",
        )

    def list_deployment_logs(
        self,
        deployment_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """List deployment log batches."""

        return self.request_json("GET", f"/v2/deployments/{deployment_id}/logs", query=query)

    def list_revision_logs(
        self,
        deployment_id: str,
        revision_id: str,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """List logs for one deployment revision."""

        return self.request_json(
            "GET",
            f"/v2/deployments/{deployment_id}/revisions/{revision_id}/logs",
            query=query,
        )

    def list_deployment_log_entries(
        self,
        *,
        deployment_id: str,
        revision_id: str | None = None,
        log_type: str = "DEPLOY",
        start_time: str | None = None,
        end_time: str | None = None,
        sort_order: str | None = None,
    ) -> Any:
        """List individual build/deploy log entries across a deployment."""

        query = {
            key: value
            for key, value in {
                "deployment_id": deployment_id,
                "revision_id": revision_id,
                "log_type": log_type,
                "start_time": start_time,
                "end_time": end_time,
                "sort_order": sort_order,
            }.items()
            if value is not None
        }
        return self.request_json("GET", "/v2/deployment-logs", query=query)

    def patch_deployment_resource_tiers(
        self,
        deployment_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Update deployment resource-tier settings."""

        return self.request_json(
            "PATCH",
            f"/v2/deployments/{deployment_id}/resource-tiers",
            body=dict(body),
        )

    def patch_deployment_tier(
        self,
        deployment_id: str,
        *,
        body: Mapping[str, Any],
    ) -> Any:
        """Update the deployment tier."""

        return self.request_json(
            "PATCH",
            f"/v2/deployments/{deployment_id}/deployment-tier",
            body=dict(body),
        )

    def get_free_deployment_count(self) -> Any:
        """Return the workspace's remaining free deployment count."""

        return self.request_json("GET", "/v2/deployments/free-count")

    def create_listener(self, *, body: Mapping[str, Any]) -> Any:
        """Create a LangSmith v2 listener."""

        return self.request_json("POST", "/v2/listeners", body=dict(body))

    def list_listeners(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List LangSmith v2 listeners with native filters."""

        return self.request_json("GET", "/v2/listeners", query=query)

    def get_listener(self, listener_id: str) -> Any:
        """Fetch one LangSmith v2 listener."""

        return self.request_json("GET", f"/v2/listeners/{listener_id}")

    def patch_listener(self, listener_id: str, *, body: Mapping[str, Any]) -> Any:
        """Patch one LangSmith v2 listener."""

        return self.request_json("PATCH", f"/v2/listeners/{listener_id}", body=dict(body))

    def delete_listener(self, listener_id: str) -> Any:
        """Delete one LangSmith v2 listener."""

        return self.request_json("DELETE", f"/v2/listeners/{listener_id}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None,
        body: Any | None,
        headers: Mapping[str, str] | None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        request_headers = {"X-Tenant-Id": self.tenant_id, **(headers or {})}
        return super()._request(
            method,
            path,
            query=query,
            body=body,
            headers=request_headers,
        )


def _default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, Mapping[str, str], bytes]:
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers.items()), response.read()
