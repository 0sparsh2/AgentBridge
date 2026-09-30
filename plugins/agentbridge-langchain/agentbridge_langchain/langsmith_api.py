"""Small dependency-free transport for LangSmith deployment/API endpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from urllib.parse import urlencode
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

    def get_thread(self, thread_id: str) -> Any:
        """Fetch a LangSmith/LangGraph deployment thread."""

        return self.request_json("GET", f"/threads/{thread_id}")

    def create_thread(self, *, metadata: Mapping[str, Any] | None = None) -> Any:
        """Create a thread for a deployed graph or Agent Server runtime."""

        return self.request_json("POST", "/threads", body={"metadata": dict(metadata or {})})

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


def _default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, Mapping[str, str], bytes]:
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers.items()), response.read()
