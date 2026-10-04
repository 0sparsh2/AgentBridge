"""Dependency-free transport for Langfuse API operations."""

from __future__ import annotations

import base64
import asyncio
import json
import os
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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
        raw = self._request(method, path, query=kwargs.pop("query", None), headers=headers, **kwargs)[2]
        for line in raw.decode("utf-8").splitlines():
            if line.startswith("data:"):
                payload = line.removeprefix("data:").strip()
                if payload and payload != "[DONE]":
                    yield json.loads(payload)

    def iter_observations(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through all Observations API v2 pages using its cursor contract."""

        page_query = dict(query or {})
        while True:
            page = self.list_observations(query=page_query)
            if not isinstance(page, Mapping):
                return
            for item in page.get("data", []) or []:
                if isinstance(item, dict):
                    yield item
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

    def iter_scores_v3(
        self,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through all Scores API v3 pages using its cursor contract."""

        page_query = dict(query or {})
        while True:
            page = self.list_scores_v3(query=page_query)
            if not isinstance(page, Mapping):
                return
            for item in page.get("data", []) or []:
                if isinstance(item, dict):
                    yield item
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor or cursor == page_query.get("cursor"):
                return
            page_query["cursor"] = cursor

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

    def list_scores_v3(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Query typed Langfuse Scores API v3."""

        return self.request_json("GET", "/api/public/v3/scores", query=query)

    def query_metrics(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """Query Langfuse Metrics API v2."""

        return self.request_json("GET", "/api/public/v2/metrics", query=query)

    def list_experiments(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List hosted experiment runs through Langfuse's current API."""

        return self.request_json("GET", "/api/public/experiments", query=query)

    def list_experiment_items(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List experiment inputs, outputs, expected outputs, and scores."""

        return self.request_json("GET", "/api/public/experiment-items", query=query)

    def list_prompts(self, *, query: Mapping[str, Any] | None = None) -> Any:
        """List prompts through Langfuse Prompt Management API v2."""

        return self.request_json("GET", "/api/public/v2/prompts", query=query)

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
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> Any:
        """Attach a numeric score to a trace using the public API contract."""

        body: dict[str, Any] = {"traceId": trace_id, "name": name, "value": value}
        if comment is not None:
            body["comment"] = comment
        return self.request_json("POST", "/api/public/scores", body=body)

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
        return self._transport(method.upper(), url, request_headers, encoded_body)


def _default_transport(method: str, url: str, headers: dict[str, str], body: bytes | None):
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers.items()), response.read()
