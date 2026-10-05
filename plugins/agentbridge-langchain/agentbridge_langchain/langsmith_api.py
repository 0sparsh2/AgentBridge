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

    def delete_thread(self, thread_id: str) -> Any:
        """Delete a deployment thread."""

        return self.request_json("DELETE", f"/threads/{thread_id}")

    def thread_history(self, thread_id: str, *, body: Mapping[str, Any] | None = None) -> Any:
        """Read a deployment thread history."""

        return self.request_json("POST", f"/threads/{thread_id}/history", body=dict(body or {}))

    def resolve_interrupt(self, thread_id: str) -> Any:
        """Resolve a paused human-in-the-loop thread."""

        return self.request_json("POST", f"/threads/{thread_id}/resolve-interrupt")

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
    ) -> Any:
        """Start a non-streaming or streaming thread run."""

        body = {"assistant_id": assistant_id, "input": input}
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
    ) -> Any:
        """Attach user or evaluator feedback to a LangSmith run."""

        body: dict[str, Any] = {"run_id": run_id, "key": key}
        optional = {
            "score": score,
            "value": value,
            "comment": comment,
            "correction": correction,
            "source_info": dict(source_info) if source_info is not None else None,
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

    def get_free_deployment_count(self) -> Any:
        """Return the workspace's remaining free deployment count."""

        return self.request_json("GET", "/v2/deployments/free-count")

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
