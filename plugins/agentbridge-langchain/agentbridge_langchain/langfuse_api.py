"""Dependency-free transport for Langfuse API operations."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Iterator, Mapping
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

    def stream_events(self, method: str, path: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Parse SSE data for ingestion or export endpoints that stream events."""

        headers = {"Accept": "text/event-stream", **(kwargs.pop("headers", {}) or {})}
        raw = self._request(method, path, query=kwargs.pop("query", None), headers=headers, **kwargs)[2]
        for line in raw.decode("utf-8").splitlines():
            if line.startswith("data:"):
                payload = line.removeprefix("data:").strip()
                if payload and payload != "[DONE]":
                    yield json.loads(payload)

    def ingest(self, events: list[dict[str, Any]]) -> Any:
        """Submit native Langfuse observations/events without hiding the payload shape."""

        return self.request_json("POST", "/api/public/ingestion", body={"batch": events})

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

    def _request(self, method: str, path: str, *, query: Mapping[str, Any] | None, body: Any | None, headers: Mapping[str, str] | None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        query_string = urlencode([(key, value) for key, value in (query or {}).items() if value is not None], doseq=True)
        if query_string:
            url = f"{url}?{query_string}"
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        request_headers = {"Accept": "application/json", "Authorization": f"Basic {token}", **(headers or {})}
        raw_body = None
        if body is not None:
            raw_body = json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        return self._transport(method.upper(), url, request_headers, raw_body)


def _default_transport(method: str, url: str, headers: dict[str, str], body: bytes | None):
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers.items()), response.read()
