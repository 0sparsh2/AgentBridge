"""Dependency-free transport for Langfuse API operations."""

from __future__ import annotations

import base64
import asyncio
import json
import os
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import aclosing
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ._sse import async_events, response_events, stream_http


Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Mapping[str, str], bytes]]


class LangfuseAPIClient:
    """Generic authenticated JSON client for Langfuse control/data APIs."""

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str = "https://cloud.langfuse.com",
        transport: Transport | None = None,
    ) -> None:
        self.public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        if not self.public_key or not self.secret_key:
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required.")
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _default_transport
        self._stream_transport = transport or stream_http

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        status, response_headers, raw = self._request(
            method, path, query=query, body=body, headers=headers
        )
        if status >= 400:
            raise RuntimeError(f"Langfuse API {status} for {method} {path}: {raw[:500]!r}")
        if not raw:
            return None
        content_type = response_headers.get("content-type", "")
        if "json" not in content_type and not raw.lstrip().startswith((b"{", b"[")):
            return raw
        return json.loads(raw)

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        body: bytes | bytearray | memoryview | Any | None = None,
        content_type: str = "application/json",
        query: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Call an endpoint with a caller-owned body encoding."""

        raw_body = body
        if body is not None and not isinstance(body, (bytes, bytearray, memoryview)):
            raw_body = json.dumps(body).encode("utf-8")
        status, response_headers, raw = self._request(
            method,
            path,
            query=query,
            body=None,
            raw_body=bytes(raw_body) if raw_body is not None else None,
            headers={"Content-Type": content_type, **(headers or {})},
        )
        if status >= 400:
            raise RuntimeError(f"Langfuse API {status} for {method} {path}: {raw[:500]!r}")
        if not raw:
            return None
        response_content_type = response_headers.get("content-type", "")
        if "json" not in response_content_type and not raw.lstrip().startswith((b"{", b"[")):
            return raw
        return json.loads(raw)

    def stream_events(self, method: str, path: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Parse SSE data for ingestion or export endpoints that stream events."""

        headers = {"Accept": "text/event-stream", **(kwargs.pop("headers", {}) or {})}
        status, _response_headers, raw = self._request(
            method,
            path,
            query=kwargs.pop("query", None),
            body=kwargs.pop("body", None),
            headers=headers,
            stream=True,
            **kwargs,
        )
        if status >= 400:
            raise RuntimeError(f"Langfuse API {status} for {method} {path}: {raw[:500]!r}")
        yield from response_events(raw)

    def iter_observations(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through all Observations API v2 pages using its cursor contract."""

        yield from _iter_cursor_pages(self.list_observations, query=query)

    def iter_scores_v3(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through all Scores API v3 pages using its cursor contract."""

        yield from _iter_cursor_pages(self.list_scores_v3, query=query)

    async def arequest_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Async facade for arbitrary Langfuse JSON endpoints."""

        return await asyncio.to_thread(
            self.request_json,
            method,
            path,
            query=query,
            body=body,
            headers=headers,
        )

    async def arequest_raw(
        self,
        method: str,
        path: str,
        *,
        body: bytes | bytearray | memoryview | Any | None = None,
        content_type: str = "application/json",
        query: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Async facade for endpoints with caller-owned body encoding."""

        return await asyncio.to_thread(
            self.request_raw,
            method,
            path,
            body=body,
            content_type=content_type,
            query=query,
            headers=headers,
        )

    async def aingest_otlp(
        self,
        payload: bytes | bytearray | memoryview | Mapping[str, Any],
        *,
        content_type: str = "application/json",
        ingestion_version: str = "4",
    ) -> Any:
        """Async OTLP trace ingestion helper."""

        return await self.arequest_raw(
            "POST",
            "/api/public/otel/v1/traces",
            body=payload,
            content_type=content_type,
            headers={"x-langfuse-ingestion-version": ingestion_version},
        )

    async def alist_observations(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query facade for Langfuse Observations API v2."""

        return await self.arequest_json("GET", "/api/public/v2/observations", query=query)

    async def aget_observation(
        self,
        observation_id: str,
        *,
        from_start_time: str,
        to_start_time: str,
        fields: str | None = None,
    ) -> Any:
        """Async current-v4 observation lookup preserving the filtered page shape."""

        filter_value = json.dumps(
            [{"type": "string", "column": "id", "operator": "=", "value": observation_id}],
            separators=(",", ":"),
        )
        return await self.alist_observations(
            query={
                "filter": filter_value,
                "fromStartTime": from_start_time,
                "toStartTime": to_start_time,
                "fields": fields,
            }
        )

    async def alist_scores_v3(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query facade for typed Langfuse Scores API v3."""

        return await self.arequest_json("GET", "/api/public/v3/scores", query=query)

    async def aiter_scores_v3(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Export typed Scores v3 records, preserving filters and detecting cursor cycles."""

        async for item in _aiter_cursor_pages(self.alist_scores_v3, query=query):
            yield item

    async def aget_score(self, score_id: str, *, fields: str | None = None) -> Any:
        """Async current-v4 score lookup."""

        return await self.alist_scores_v3(query={"id": score_id, "fields": fields})

    async def aquery_metrics(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query for Langfuse Metrics API v2."""

        return await self.arequest_json("GET", "/api/public/v2/metrics", query=query)

    async def alist_experiments(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query facade for hosted experiment runs."""

        return await self.arequest_json("GET", "/api/public/experiments", query=query)

    async def aiter_experiments(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for hosted experiments."""

        async for item in _aiter_cursor_pages(self.alist_experiments, query=query):
            yield item

    async def alist_experiment_items(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query facade for experiment inputs, outputs, and scores."""

        return await self.arequest_json("GET", "/api/public/experiment-items", query=query)

    async def aiter_experiment_items(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for experiment items."""

        async for item in _aiter_cursor_pages(self.alist_experiment_items, query=query):
            yield item

    async def alist_prompts(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async list for Langfuse Prompt Management API v2."""

        return await self.arequest_json("GET", "/api/public/v2/prompts", query=query)

    async def aiter_prompts(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async numbered-page iterator preserving native filters."""

        async for item in _aiter_numbered_pages(self.alist_prompts, query=query):
            yield item

    async def alist_datasets(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query facade for Langfuse Dataset API v2."""

        return await self.arequest_json("GET", "/api/public/v2/datasets", query=query)

    async def aiter_datasets(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async numbered-page iterator preserving native filters."""

        async for item in _aiter_numbered_pages(self.alist_datasets, query=query):
            yield item

    async def alist_dataset_items(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Async query facade for versioned dataset items."""

        return await self.arequest_json("GET", "/api/public/dataset-items", query=query)

    async def aiter_dataset_items(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async numbered-page iterator preserving native filters."""

        async for item in _aiter_numbered_pages(self.alist_dataset_items, query=query):
            yield item

    async def aiter_observations(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async cursor iterator for Observations API v2."""

        async for item in _aiter_cursor_pages(self.alist_observations, query=query):
            yield item

    async def aget_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
        prompt_type: str | None = None,
    ) -> Any:
        """Async fetch for a Langfuse prompt version or label."""

        query = {
            key: value
            for key, value in {
                "label": label,
                "version": version,
                "type": prompt_type,
            }.items()
            if value is not None
        }
        return await self.arequest_json("GET", f"/api/public/v2/prompts/{name}", query=query)

    async def acreate_prompt(
        self,
        *,
        name: str,
        prompt: Any,
        prompt_type: str,
        labels: list[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async create for a Langfuse prompt version."""

        body: dict[str, Any] = {"name": name, "prompt": prompt, "type": prompt_type}
        if labels is not None:
            body["labels"] = list(labels)
        if config is not None:
            body["config"] = dict(config)
        return await self.arequest_json("POST", "/api/public/v2/prompts", body=body)

    async def acreate_score(
        self,
        *,
        name: str,
        value: Any,
        trace_id: str | None = None,
        session_id: str | None = None,
        observation_id: str | None = None,
        dataset_run_id: str | None = None,
        score_id: str | None = None,
        data_type: str | None = None,
        config_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        description: str | None = None,
        source: str | None = None,
        environment: str | None = None,
        timestamp: str | None = None,
        comment: str | None = None,
    ) -> Any:
        """Async create/upsert for a typed Langfuse score."""

        body: dict[str, Any] = {"name": name, "value": value}
        optional = {
            "traceId": trace_id,
            "sessionId": session_id,
            "observationId": observation_id,
            "datasetRunId": dataset_run_id,
            "id": score_id,
            "dataType": data_type,
            "configId": config_id,
            "metadata": dict(metadata) if metadata is not None else None,
            "description": description,
            "source": source,
            "environment": environment,
            "timestamp": timestamp,
            "comment": comment,
        }
        body.update({key: item for key, item in optional.items() if item is not None})
        return await self.arequest_json("POST", "/api/public/scores", body=body)

    async def acreate_dataset(self, *, name: str, description: str | None = None) -> Any:
        """Async create for a Langfuse dataset."""

        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        return await self.arequest_json("POST", "/api/public/v2/datasets", body=body)

    async def aget_dataset(self, name: str, *, version: str | None = None) -> Any:
        """Async fetch for a Langfuse dataset."""

        query = {"version": version} if version is not None else None
        return await self.arequest_json("GET", f"/api/public/v2/datasets/{name}", query=query)

    async def adelete_dataset(self, name: str) -> Any:
        """Async delete a Langfuse dataset and its items."""

        return await self.arequest_json("DELETE", f"/api/public/v2/datasets/{name}")

    async def acreate_dataset_item(
        self,
        *,
        dataset_name: str,
        input: Any = None,
        expected_output: Any = None,
        item_id: str | None = None,
        metadata: Any = None,
        source_trace_id: str | None = None,
        source_observation_id: str | None = None,
        status: str | None = None,
    ) -> Any:
        """Async create/upsert for a versioned Langfuse dataset item."""

        body: dict[str, Any] = {"datasetName": dataset_name}
        optional = {
            "input": input,
            "expectedOutput": expected_output,
            "id": item_id,
            "metadata": metadata,
            "sourceTraceId": source_trace_id,
            "sourceObservationId": source_observation_id,
            "status": status,
        }
        body.update({key: item for key, item in optional.items() if item is not None})
        return await self.arequest_json("POST", "/api/public/dataset-items", body=body)

    async def aupdate_dataset_item(self, item_id: str, *, body: Mapping[str, Any]) -> Any:
        """Async upsert an update for a versioned Langfuse dataset item."""

        return await self.acreate_dataset_item(
            dataset_name=str(body.get("datasetName", "")),
            input=body.get("input"),
            expected_output=body.get("expectedOutput"),
            item_id=item_id,
            metadata=body.get("metadata"),
            source_trace_id=body.get("sourceTraceId"),
            source_observation_id=body.get("sourceObservationId"),
            status=body.get("status"),
        )

    async def aget_trace(self, trace_id: str) -> Any:
        """Async fetch for one legacy Langfuse trace payload."""

        return await self.arequest_json("GET", f"/api/public/traces/{trace_id}")

    async def adelete_trace(self, trace_id: str) -> Any:
        """Async delete for one trace and its observations/scores."""

        return await self.arequest_json("DELETE", f"/api/public/traces/{trace_id}")

    async def adelete_traces(self, *, trace_ids: list[str]) -> Any:
        """Async batch delete for traces."""

        return await self.arequest_json(
            "POST",
            "/api/public/traces/delete",
            body={"traceIds": trace_ids},
        )

    async def aget_dataset_item(self, item_id: str) -> Any:
        """Async fetch for one dataset item."""

        return await self.arequest_json("GET", f"/api/public/dataset-items/{item_id}")

    async def adelete_dataset_item(self, item_id: str) -> Any:
        """Async delete for one dataset item and its experiment items."""

        return await self.arequest_json("DELETE", f"/api/public/dataset-items/{item_id}")

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

        async with aclosing(async_events(self.stream_events(
            method, path, query=query, body=body, headers=headers
        ))) as events:
            async for event in events:
                yield event

    def ingest(self, events: list[dict[str, Any]]) -> Any:
        """Submit native Langfuse observations/events without hiding the payload shape."""

        warnings.warn(
            "Langfuse legacy ingestion is deprecated; use ingest_otlp() for trace telemetry.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.request_json("POST", "/api/public/ingestion", body={"batch": events})

    def ingest_otlp(
        self,
        payload: bytes | bytearray | memoryview | Mapping[str, Any],
        *,
        content_type: str = "application/json",
        ingestion_version: str = "4",
    ) -> Any:
        """Submit traces through Langfuse's current OTLP/HTTP ingestion endpoint."""

        return self.request_raw(
            "POST",
            "/api/public/otel/v1/traces",
            body=payload,
            content_type=content_type,
            headers={"x-langfuse-ingestion-version": ingestion_version},
        )

    def list_observations(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Query Langfuse Observations API v2."""

        return self.request_json("GET", "/api/public/v2/observations", query=query)

    def list_trace_observations(
        self,
        trace_id: str,
        *,
        from_start_time: str,
        to_start_time: str,
        fields: str | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Retrieve current v4 observations for one trace in a bounded time window."""

        request_query = {
            "traceId": trace_id,
            "fromStartTime": from_start_time,
            "toStartTime": to_start_time,
            "fields": fields,
            **dict(query or {}),
        }
        return self.list_observations(query=request_query)

    def get_observation(
        self,
        observation_id: str,
        *,
        from_start_time: str,
        to_start_time: str,
        fields: str | None = None,
    ) -> Any:
        """Find one observation through the v2 filter contract.

        Langfuse v4 intentionally has no observation-by-ID route. The API returns a
        page, so this method preserves that native response shape rather than
        guessing whether a missing row means not-found or eventual consistency.
        """

        filter_value = json.dumps(
            [{"type": "string", "column": "id", "operator": "=", "value": observation_id}],
            separators=(",", ":"),
        )
        return self.list_observations(
            query={
                "filter": filter_value,
                "fromStartTime": from_start_time,
                "toStartTime": to_start_time,
                "fields": fields,
            }
        )

    def list_scores_v3(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Query typed Langfuse Scores API v3."""

        return self.request_json("GET", "/api/public/v3/scores", query=query)

    def get_score(self, score_id: str, *, fields: str | None = None) -> Any:
        """Fetch one score through the current Scores v3 query endpoint."""

        return self.list_scores_v3(query={"id": score_id, "fields": fields})

    def query_metrics(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Query Langfuse Metrics API v2."""

        return self.request_json("GET", "/api/public/v2/metrics", query=query)

    def list_experiments(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List hosted experiment runs through Langfuse's current API."""

        return self.request_json("GET", "/api/public/experiments", query=query)

    def iter_experiments(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through cursor-paginated current experiment responses."""

        yield from _iter_cursor_pages(self.list_experiments, query=query)

    def list_experiment_items(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List experiment inputs, outputs, expected outputs, and scores."""

        return self.request_json("GET", "/api/public/experiment-items", query=query)

    def iter_experiment_items(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through cursor-paginated current experiment-item responses."""

        yield from _iter_cursor_pages(self.list_experiment_items, query=query)

    def list_prompts(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List prompts through Langfuse Prompt Management API v2."""

        return self.request_json("GET", "/api/public/v2/prompts", query=query)

    def iter_prompts(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate numbered pages while preserving native filters."""

        yield from _iter_numbered_pages(self.list_prompts, query=query)

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
        prompt_type: str | None = None,
    ) -> Any:
        """Fetch a prompt by deployment label or immutable version."""

        query = {
            key: value
            for key, value in {
                "label": label,
                "version": version,
                "type": prompt_type,
            }.items()
            if value is not None
        }
        return self.request_json("GET", f"/api/public/v2/prompts/{name}", query=query)

    def create_prompt(
        self,
        *,
        name: str,
        prompt: Any,
        prompt_type: str,
        labels: list[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Create a new version of a Langfuse text or chat prompt."""

        body: dict[str, Any] = {"name": name, "prompt": prompt, "type": prompt_type}
        if labels is not None:
            body["labels"] = list(labels)
        if config is not None:
            body["config"] = dict(config)
        return self.request_json("POST", "/api/public/v2/prompts", body=body)

    def create_score(
        self,
        *,
        name: str,
        value: Any,
        trace_id: str | None = None,
        session_id: str | None = None,
        observation_id: str | None = None,
        dataset_run_id: str | None = None,
        score_id: str | None = None,
        data_type: str | None = None,
        config_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        description: str | None = None,
        source: str | None = None,
        environment: str | None = None,
        timestamp: str | None = None,
        comment: str | None = None,
    ) -> Any:
        """Create or upsert a typed score using the native public API contract.

        The target is intentionally optional: Langfuse accepts scores attached to a
        trace, session, observation, or dataset run, and also supports standalone
        scores. ``value`` is left typed as ``Any`` so numeric, boolean, categorical,
        and text scores retain their native representation.
        """

        body: dict[str, Any] = {"name": name, "value": value}
        optional = {
            "traceId": trace_id,
            "sessionId": session_id,
            "observationId": observation_id,
            "datasetRunId": dataset_run_id,
            "id": score_id,
            "dataType": data_type,
            "configId": config_id,
            "metadata": dict(metadata) if metadata is not None else None,
            "description": description,
            "source": source,
            "environment": environment,
            "timestamp": timestamp,
            "comment": comment,
        }
        body.update({key: item for key, item in optional.items() if item is not None})
        return self.request_json("POST", "/api/public/scores", body=body)

    def list_score_configs(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> Any:
        """list score configs through the native public API."""

        return self.request_json(
            "GET", "/api/public/score-configs", query=query,
        )

    async def alist_score_configs(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list score configs through the native public API."""

        return await self.arequest_json(
            "GET", "/api/public/score-configs", query=query,
        )

    def get_score_config(
        self, config_id: str,
    ) -> Any:
        """get score config through the native public API."""

        return self.request_json(
            "GET", f"/api/public/score-configs/{_path_id(config_id)}",
        )

    async def aget_score_config(
        self, config_id: str,
    ) -> Any:
        """Async get score config through the native public API."""

        return await self.arequest_json(
            "GET", f"/api/public/score-configs/{_path_id(config_id)}",
        )

    def create_score_config(
        self, *, name: str, data_type: str,
        categories: list[Mapping[str, Any]] | None = None,
        min_value: float | None = None, max_value: float | None = None,
        description: str | None = None,
    ) -> Any:
        """create score config through the native public API."""

        body = _score_config_body(
            name=name, data_type=data_type, categories=categories,
            min_value=min_value, max_value=max_value, description=description,
        )
        return self.request_json("POST", "/api/public/score-configs", body=body)

    async def acreate_score_config(
        self, *, name: str, data_type: str,
        categories: list[Mapping[str, Any]] | None = None,
        min_value: float | None = None, max_value: float | None = None,
        description: str | None = None,
    ) -> Any:
        """Async create score config through the native public API."""

        body = _score_config_body(
            name=name, data_type=data_type, categories=categories,
            min_value=min_value, max_value=max_value, description=description,
        )
        return await self.arequest_json("POST", "/api/public/score-configs", body=body)

    def update_score_config(
        self, config_id: str, *, body: Mapping[str, Any],
    ) -> Any:
        """update score config through the native public API."""

        return self.request_json(
            "PATCH", f"/api/public/score-configs/{_path_id(config_id)}", body=dict(body),
        )

    async def aupdate_score_config(
        self, config_id: str, *, body: Mapping[str, Any],
    ) -> Any:
        """Async update score config through the native public API."""

        return await self.arequest_json(
            "PATCH", f"/api/public/score-configs/{_path_id(config_id)}", body=dict(body),
        )

    def list_annotation_queues(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> Any:
        """list annotation queues through the native public API."""

        return self.request_json(
            "GET", "/api/public/annotation-queues", query=query,
        )

    async def alist_annotation_queues(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list annotation queues through the native public API."""

        return await self.arequest_json(
            "GET", "/api/public/annotation-queues", query=query,
        )

    def get_annotation_queue(
        self, queue_id: str,
    ) -> Any:
        """get annotation queue through the native public API."""

        return self.request_json(
            "GET", f"/api/public/annotation-queues/{_path_id(queue_id)}",
        )

    async def aget_annotation_queue(
        self, queue_id: str,
    ) -> Any:
        """Async get annotation queue through the native public API."""

        return await self.arequest_json(
            "GET", f"/api/public/annotation-queues/{_path_id(queue_id)}",
        )

    def create_annotation_queue(
        self, *, name: str, score_config_ids: list[str], description: str | None = None,
    ) -> Any:
        """create annotation queue through the native public API."""

        return self.request_json(
            "POST", "/api/public/annotation-queues", body={"name": name, "scoreConfigIds": list(score_config_ids),
                  **({"description": description} if description is not None else {})},
        )

    async def acreate_annotation_queue(
        self, *, name: str, score_config_ids: list[str], description: str | None = None,
    ) -> Any:
        """Async create annotation queue through the native public API."""

        return await self.arequest_json(
            "POST", "/api/public/annotation-queues", body={"name": name, "scoreConfigIds": list(score_config_ids),
                  **({"description": description} if description is not None else {})},
        )

    def list_annotation_queue_items(
        self, queue_id: str, *, query: Mapping[str, Any] | None = None,
    ) -> Any:
        """list annotation queue items through the native public API."""

        return self.request_json(
            "GET", f"/api/public/annotation-queues/{_path_id(queue_id)}/items", query=query,
        )

    async def alist_annotation_queue_items(
        self, queue_id: str, *, query: Mapping[str, Any] | None = None,
    ) -> Any:
        """Async list annotation queue items through the native public API."""

        return await self.arequest_json(
            "GET", f"/api/public/annotation-queues/{_path_id(queue_id)}/items", query=query,
        )

    def get_annotation_queue_item(
        self, queue_id: str, item_id: str,
    ) -> Any:
        """get annotation queue item through the native public API."""

        return self.request_json(
            "GET", f"/api/public/annotation-queues/{_path_id(queue_id)}/items/{_path_id(item_id)}",
        )

    async def aget_annotation_queue_item(
        self, queue_id: str, item_id: str,
    ) -> Any:
        """Async get annotation queue item through the native public API."""

        return await self.arequest_json(
            "GET", f"/api/public/annotation-queues/{_path_id(queue_id)}/items/{_path_id(item_id)}",
        )

    def create_annotation_queue_item(
        self, queue_id: str, *, object_id: str, object_type: str, status: str | None = None,
    ) -> Any:
        """create annotation queue item through the native public API."""

        return self.request_json(
            "POST", f"/api/public/annotation-queues/{_path_id(queue_id)}/items", body={"objectId": object_id, "objectType": object_type,
                  **({"status": status} if status is not None else {})},
        )

    async def acreate_annotation_queue_item(
        self, queue_id: str, *, object_id: str, object_type: str, status: str | None = None,
    ) -> Any:
        """Async create annotation queue item through the native public API."""

        return await self.arequest_json(
            "POST", f"/api/public/annotation-queues/{_path_id(queue_id)}/items", body={"objectId": object_id, "objectType": object_type,
                  **({"status": status} if status is not None else {})},
        )

    def update_annotation_queue_item(
        self, queue_id: str, item_id: str, *, body: Mapping[str, Any],
    ) -> Any:
        """update annotation queue item through the native public API."""

        return self.request_json(
            "PATCH", f"/api/public/annotation-queues/{_path_id(queue_id)}/items/{_path_id(item_id)}", body=dict(body),
        )

    async def aupdate_annotation_queue_item(
        self, queue_id: str, item_id: str, *, body: Mapping[str, Any],
    ) -> Any:
        """Async update annotation queue item through the native public API."""

        return await self.arequest_json(
            "PATCH", f"/api/public/annotation-queues/{_path_id(queue_id)}/items/{_path_id(item_id)}", body=dict(body),
        )

    def delete_annotation_queue_item(
        self, queue_id: str, item_id: str,
    ) -> Any:
        """delete annotation queue item through the native public API."""

        return self.request_json(
            "DELETE", f"/api/public/annotation-queues/{_path_id(queue_id)}/items/{_path_id(item_id)}",
        )

    async def adelete_annotation_queue_item(
        self, queue_id: str, item_id: str,
    ) -> Any:
        """Async delete annotation queue item through the native public API."""

        return await self.arequest_json(
            "DELETE", f"/api/public/annotation-queues/{_path_id(queue_id)}/items/{_path_id(item_id)}",
        )

    def create_annotation_queue_assignment(
        self, queue_id: str, *, user_id: str,
    ) -> Any:
        """create annotation queue assignment through the native public API."""

        return self.request_json(
            "POST", f"/api/public/annotation-queues/{_path_id(queue_id)}/assignments", body={"userId": user_id},
        )

    async def acreate_annotation_queue_assignment(
        self, queue_id: str, *, user_id: str,
    ) -> Any:
        """Async create annotation queue assignment through the native public API."""

        return await self.arequest_json(
            "POST", f"/api/public/annotation-queues/{_path_id(queue_id)}/assignments", body={"userId": user_id},
        )

    def delete_annotation_queue_assignment(
        self, queue_id: str, *, user_id: str,
    ) -> Any:
        """delete annotation queue assignment through the native public API."""

        return self.request_json(
            "DELETE", f"/api/public/annotation-queues/{_path_id(queue_id)}/assignments", body={"userId": user_id},
        )

    async def adelete_annotation_queue_assignment(
        self, queue_id: str, *, user_id: str,
    ) -> Any:
        """Async delete annotation queue assignment through the native public API."""

        return await self.arequest_json(
            "DELETE", f"/api/public/annotation-queues/{_path_id(queue_id)}/assignments", body={"userId": user_id},
        )

    def iter_score_configs(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate native numbered pages of score configs."""

        yield from _iter_numbered_pages(self.list_score_configs, query=query)

    async def aiter_score_configs(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate native numbered pages of score configs."""

        async for item in _aiter_numbered_pages(self.alist_score_configs, query=query):
            yield item

    def iter_annotation_queues(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate native numbered pages of annotation queues."""

        yield from _iter_numbered_pages(self.list_annotation_queues, query=query)

    async def aiter_annotation_queues(
        self, *, query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate native numbered pages of annotation queues."""

        async for item in _aiter_numbered_pages(self.alist_annotation_queues, query=query):
            yield item

    def iter_annotation_queue_items(
        self, queue_id: str, *, query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate queue items while preserving status and pagination filters."""

        def fetch(*, query):
            return self.list_annotation_queue_items(queue_id, query=query)

        yield from _iter_numbered_pages(fetch, query=query)

    async def aiter_annotation_queue_items(
        self, queue_id: str, *, query: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate queue items while preserving status and pagination filters."""

        def fetch(*, query):
            return self.alist_annotation_queue_items(queue_id, query=query)

        async for item in _aiter_numbered_pages(fetch, query=query):
            yield item


    def get_trace(self, trace_id: str) -> Any:
        """Retrieve one legacy trace payload for compatibility.

        New extraction workflows should use ``list_observations`` with a
        ``traceId`` filter, matching Langfuse's current v4 data model.
        """

        return self.request_json("GET", f"/api/public/traces/{trace_id}")

    def delete_trace(self, trace_id: str) -> Any:
        """Delete one trace and its observations/scores."""

        return self.request_json("DELETE", f"/api/public/traces/{trace_id}")

    def delete_traces(self, *, trace_ids: list[str]) -> Any:
        """Delete multiple traces through the native batch payload."""

        return self.request_json("POST", "/api/public/traces/delete", body={"traceIds": trace_ids})

    def create_dataset(self, *, name: str, description: str | None = None) -> Any:
        """Create a Langfuse dataset for framework-neutral evaluations."""

        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        return self.request_json("POST", "/api/public/v2/datasets", body=body)

    def list_datasets(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List datasets through the current Dataset API v2."""

        return self.request_json("GET", "/api/public/v2/datasets", query=query)

    def iter_datasets(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate numbered pages while preserving native filters."""

        yield from _iter_numbered_pages(self.list_datasets, query=query)

    def get_dataset(self, name: str, *, version: str | None = None) -> Any:
        """Fetch a dataset, optionally at a historical item-version timestamp."""

        query = {"version": version} if version is not None else None
        return self.request_json("GET", f"/api/public/v2/datasets/{name}", query=query)

    def delete_dataset(self, name: str) -> Any:
        """Delete a dataset and its items."""

        return self.request_json("DELETE", f"/api/public/v2/datasets/{name}")

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        input: Any = None,
        expected_output: Any = None,
        item_id: str | None = None,
        metadata: Any = None,
        source_trace_id: str | None = None,
        source_observation_id: str | None = None,
        status: str | None = None,
    ) -> Any:
        """Create or upsert one versioned item in a Langfuse dataset."""

        body: dict[str, Any] = {"datasetName": dataset_name}
        optional = {
            "input": input,
            "expectedOutput": expected_output,
            "id": item_id,
            "metadata": metadata,
            "sourceTraceId": source_trace_id,
            "sourceObservationId": source_observation_id,
            "status": status,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self.request_json("POST", "/api/public/dataset-items", body=body)

    def update_dataset_item(self, item_id: str, *, body: Mapping[str, Any]) -> Any:
        """Upsert an item update using the native item identifier."""

        return self.create_dataset_item(
            dataset_name=str(body.get("datasetName", "")),
            input=body.get("input"),
            expected_output=body.get("expectedOutput"),
            item_id=item_id,
            metadata=body.get("metadata"),
            source_trace_id=body.get("sourceTraceId"),
            source_observation_id=body.get("sourceObservationId"),
            status=body.get("status"),
        )

    def list_dataset_items(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List dataset items with native filters and version selection."""

        return self.request_json("GET", "/api/public/dataset-items", query=query)

    def iter_dataset_items(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate numbered pages while preserving native filters."""

        yield from _iter_numbered_pages(self.list_dataset_items, query=query)

    def get_dataset_item(self, item_id: str) -> Any:
        """Fetch one dataset item."""

        return self.request_json("GET", f"/api/public/dataset-items/{item_id}")

    def delete_dataset_item(self, item_id: str) -> Any:
        """Delete one dataset item and its associated experiment items."""

        return self.request_json("DELETE", f"/api/public/dataset-items/{item_id}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None,
        body: Any | None,
        raw_body: bytes | None = None,
        headers: Mapping[str, str] | None,
        stream: bool = False,
    ):
        url = f"{self.base_url}/{path.lstrip('/')}"
        query_string = urlencode([(key, value) for key, value in (query or {}).items() if value is not None], doseq=True)
        if query_string:
            url = f"{url}?{query_string}"
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        request_headers = {"Accept": "application/json", "Authorization": f"Basic {token}", **(headers or {})}
        encoded_body = raw_body
        if body is not None:
            encoded_body = json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        transport = self._stream_transport if stream else self._transport
        return transport(method.upper(), url, request_headers, encoded_body)


def _path_id(value: str) -> str:
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ValueError("Langfuse resource identifier must be a non-empty string.")
    return quote(value, safe="")


def _score_config_body(
    *, name: str, data_type: str, categories: list[Mapping[str, Any]] | None,
    min_value: float | None, max_value: float | None, description: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name, "dataType": data_type}
    optional = {
        "categories": [dict(category) for category in categories] if categories is not None else None,
        "minValue": min_value, "maxValue": max_value, "description": description,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def _numbered_query(query: Mapping[str, Any] | None) -> dict[str, Any]:
    page_query = dict(query or {})
    if page_query.get("cursor") is not None:
        raise ValueError("This Langfuse endpoint uses page numbers, not cursors.")
    page_query.pop("cursor", None)
    page = page_query.get("page", 1)
    if type(page) is not int or page < 1:
        raise ValueError("Langfuse page must be a positive integer.")
    return page_query


def _numbered_page(response: Any, *, requested_page: int) -> tuple[list[dict[str, Any]], int | None]:
    """Validate server pagination before declaring an export complete."""

    if not isinstance(response, Mapping) or not isinstance(response.get("data"), list):
        raise RuntimeError("Invalid Langfuse page; export is incomplete.")
    items, meta = response["data"], response.get("meta")
    if not all(isinstance(item, dict) for item in items) or not isinstance(meta, Mapping):
        raise RuntimeError("Invalid Langfuse records or metadata; export is incomplete.")
    for key in ("page", "limit", "totalItems", "totalPages"):
        minimum = 1 if key in ("page", "limit") else 0
        if type(meta.get(key)) is not int or meta[key] < minimum:
            raise RuntimeError("Invalid Langfuse page metadata; export is incomplete.")
    if meta["page"] != requested_page:
        raise RuntimeError("Langfuse returned an unexpected page; export is incomplete.")
    next_page = requested_page + 1 if requested_page < meta["totalPages"] else None
    return items, next_page


def _iter_numbered_pages(
    fetch: Callable[..., Any], *, query: Mapping[str, Any] | None,
) -> Iterator[dict[str, Any]]:
    page_query = _numbered_query(query)
    while True:
        items, next_page = _numbered_page(
            fetch(query=page_query), requested_page=page_query.get("page", 1),
        )
        yield from items
        if next_page is None:
            return
        page_query["page"] = next_page


async def _aiter_numbered_pages(
    fetch: Callable[..., Any], *, query: Mapping[str, Any] | None,
) -> AsyncIterator[dict[str, Any]]:
    page_query = _numbered_query(query)
    while True:
        items, next_page = _numbered_page(
            await fetch(query=page_query), requested_page=page_query.get("page", 1),
        )
        for item in items:
            yield item
        if next_page is None:
            return
        page_query["page"] = next_page


def _cursor_query(query: Mapping[str, Any] | None) -> dict[str, Any]:
    page_query = dict(query or {})
    cursor = page_query.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ValueError("Langfuse cursor must be an opaque string.")
    if page_query.get("page") is not None:
        raise ValueError("This Langfuse endpoint uses cursors, not page numbers.")
    page_query.pop("page", None)
    return page_query


def _iter_cursor_pages(
    fetch: Callable[..., Any], *, query: Mapping[str, Any] | None,
) -> Iterator[dict[str, Any]]:
    page_query = _cursor_query(query)
    seen_cursors = {page_query["cursor"]} if page_query.get("cursor") else set()
    while True:
        items, cursor = _cursor_page(fetch(query=page_query))
        yield from items
        if not cursor:
            return
        if cursor in seen_cursors:
            raise RuntimeError("Langfuse pagination repeated a cursor; export is incomplete.")
        seen_cursors.add(cursor)
        page_query["cursor"] = cursor


async def _aiter_cursor_pages(
    fetch: Callable[..., Any], *, query: Mapping[str, Any] | None,
) -> AsyncIterator[dict[str, Any]]:
    page_query = _cursor_query(query)
    seen_cursors = {page_query["cursor"]} if page_query.get("cursor") else set()
    while True:
        items, cursor = _cursor_page(await fetch(query=page_query))
        for item in items:
            yield item
        if not cursor:
            return
        if cursor in seen_cursors:
            raise RuntimeError("Langfuse pagination repeated a cursor; export is incomplete.")
        seen_cursors.add(cursor)
        page_query["cursor"] = cursor


def _cursor_page(page: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Reject malformed cursor pages rather than reporting truncated exports as complete."""

    if not isinstance(page, Mapping) or not isinstance(page.get("data"), list):
        raise RuntimeError("Invalid Langfuse page; export is incomplete.")
    items = page["data"]
    if not all(isinstance(item, dict) for item in items):
        raise RuntimeError("Invalid Langfuse record; export is incomplete.")
    meta = page.get("meta")
    if not isinstance(meta, Mapping):
        raise RuntimeError("Invalid Langfuse metadata; export is incomplete.")
    cursor = meta.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise RuntimeError("Invalid Langfuse cursor; export is incomplete.")
    return items, cursor


def _default_transport(method: str, url: str, headers: dict[str, str], body: bytes | None):
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers.items()), response.read()
