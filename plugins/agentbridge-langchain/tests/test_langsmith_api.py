from __future__ import annotations

import json
import asyncio

from agentbridge_langchain.langsmith_api import LangSmithAPIClient, LangSmithControlPlaneClient


def test_langsmith_api_client_preserves_arbitrary_json_endpoint():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        return 200, {"content-type": "application/json"}, b'{"ok":true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    result = client.request_json(
        "POST",
        "/v1/threads",
        query={"limit": 2, "skip": None},
        body={"metadata": {"source": "agentbridge"}},
    )

    assert result == {"ok": True}
    assert calls[0][0:2] == ("POST", "https://example.test/v1/threads?limit=2")
    assert calls[0][2]["X-Api-Key"] == "secret"
    assert json.loads(calls[0][3]) == {"metadata": {"source": "agentbridge"}}


def test_langsmith_api_client_parses_sse_data_lines():
    def transport(method, url, headers, body):
        del url, headers, body
        if method == "GET":
            return 200, {"content-type": "application/json"}, b'{"ok":true}'
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)

    assert list(client.stream_events("POST", "v1/runs/stream")) == [{"event": "update"}]


def test_langsmith_universal_call_covers_json_sse_and_async_endpoints():
    def transport(method, url, headers, body):
        del method, url, headers, body
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)
    assert list(client.call("POST", "/custom/stream", stream=True)) == [{"event": "update"}]

    async def collect():
        events = await client.acall("POST", "/custom/stream", stream=True)
        return [event async for event in events]

    assert asyncio.run(collect()) == [{"event": "update"}]


def test_langsmith_api_client_supports_async_json_and_sse_facades():
    def transport(method, url, headers, body):
        del url, headers, body
        if method == "GET":
            return 200, {"content-type": "application/json"}, b'{"ok":true}'
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)

    async def collect():
        response = await client.arequest_json("GET", "/v1/threads")
        events = [event async for event in client.astream_events("POST", "/v1/runs/stream")]
        return response, events

    response, events = asyncio.run(collect())
    assert response == {"ok": True}
    assert events == [{"event": "update"}]


def test_langsmith_thread_helpers_preserve_native_api_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if url.endswith("/runs/stream"):
            return 200, {}, b'data: {"event": "message"}\n\ndata: [DONE]\n'
        return 200, {"content-type": "application/json"}, b'{"id": "thread-1"}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.create_thread(metadata={"team": "support"})["id"] == "thread-1"
    assert client.get_thread("thread-1")["id"] == "thread-1"
    assert list(client.stream_thread_run("thread-1", assistant_id="assistant", input={"x": 1})) == [
        {"event": "message"}
    ]
    assert calls[0][2] == b'{"metadata": {"team": "support"}}'


def test_langsmith_assistant_run_and_state_helpers_preserve_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", transport=transport)
    assert client.create_assistant(graph_id="agent", config={"model": "offline"}) == {"ok": True}
    assert client.update_assistant("assistant-1", config={"prompt": "hello"}) == {"ok": True}
    assert client.create_thread_run("thread-1", assistant_id="assistant-1", input={"x": 1}) == {"ok": True}
    assert client.get_thread_state("thread-1", checkpoint_id="cp-1") == {"ok": True}
    assert client.update_thread_state("thread-1", values={"approved": True}, as_node="review") == {"ok": True}
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/assistants")
    assert calls[-1][2] == b'{"values": {"approved": true}, "as_node": "review"}'


def test_langsmith_assistant_introspection_and_thread_run_lifecycle_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if url.endswith("/join"):
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"done"}\n\ndata: [DONE]\n'
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.count_assistants(body={"graph_id": "agent"})
    client.get_assistant_graph("assistant-1", xray=2)
    client.get_assistant_schemas("assistant-1")
    client.get_assistant_subgraphs("assistant-1")
    client.get_assistant_subgraphs("assistant-1", namespace="child/ns")
    client.get_assistant_versions("assistant-1", query={"limit": 10})
    client.patch_thread("thread-1", body={"metadata": {"team": "support"}})
    client.copy_thread("thread-1")
    client.list_thread_runs("thread-1", query={"limit": 10})
    client.list_run_events("thread-1", "run-1")
    client.join_run("thread-1", "run-1")
    assert list(client.join_run_stream("thread-1", "run-1")) == [{"event": "done"}]
    client.cancel_runs("thread-1", body={"run_ids": ["run-1"]})

    assert calls[0][1].endswith("/assistants/count")
    assert calls[1][1].endswith("/assistants/assistant-1/graph?xray=2")
    assert calls[4][1].endswith("/assistants/assistant-1/subgraphs/child%2Fns")
    assert calls[7][1].endswith("/threads/thread-1/copy")
    assert calls[8][1].endswith("/threads/thread-1/runs?limit=10")
    assert calls[10][1].endswith("/threads/thread-1/runs/run-1/join")


def test_langsmith_stateless_runs_and_cron_lifecycle_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if url.endswith("/runs/stream"):
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    body = {"assistant_id": "agent", "input": {"messages": []}, "context": {"tenant": "support"}}
    client.create_background_run(body=body)
    client.create_run_wait(body=body)
    assert list(client.create_run_stream(body=body)) == [{"event": "update"}]
    client.create_run_batch(body={"runs": [body]})
    client.create_cron(body={"schedule": "0 * * * *", **body})
    client.create_thread_cron("thread-1", body={"schedule": "@hourly", **body})
    client.search_crons(body={"assistant_id": "agent"})
    client.count_crons()
    client.get_cron("cron-1")
    client.update_cron("cron-1", body={"enabled": False})
    assert client.delete_cron("cron-1") is None

    assert calls[0][1].endswith("/runs")
    assert calls[1][1].endswith("/runs/wait")
    assert calls[2][1].endswith("/runs/stream")
    assert calls[4][1].endswith("/runs/crons")
    assert calls[5][1].endswith("/threads/thread-1/runs/crons")
    assert calls[-1][1].endswith("/runs/crons/cron-1")


def test_langsmith_protocol_helpers_preserve_a2a_and_mcp_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if method == "DELETE":
            return 204, {}, b""
        return 200, {"content-type": "application/json"}, b'{"jsonrpc":"2.0","result":{}}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {}}}
    assert client.a2a_json_rpc("assistant-1", body=rpc) == {"jsonrpc": "2.0", "result": {}}
    assert client.mcp_get() == {"jsonrpc": "2.0", "result": {}}
    assert client.mcp_post(body={"jsonrpc": "2.0", "id": 2, "method": "initialize"}) == {
        "jsonrpc": "2.0",
        "result": {},
    }
    assert client.mcp_terminate() is None

    assert calls[0][0:2] == ("POST", "https://example.test/a2a/assistant-1")
    assert calls[0][2]["Accept"] == "application/json"
    assert json.loads(calls[0][3]) == rpc
    assert calls[1][0:2] == ("GET", "https://example.test/mcp/")
    assert calls[2][2]["Accept"] == "application/json"
    assert calls[3][0:2] == ("DELETE", "https://example.test/mcp/")


def test_langsmith_protocol_helpers_stream_a2a_json_rpc_events():
    def transport(method, url, headers, body):
        assert method == "POST"
        assert url.endswith("/a2a/assistant-1")
        assert headers["Accept"] == "text/event-stream"
        assert json.loads(body)["method"] == "message/stream"
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"task"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)
    assert list(client.a2a_stream("assistant-1", body={"method": "message/stream"})) == [
        {"event": "task"}
    ]


def test_langsmith_agent_server_helpers_cover_threads_runs_assistants_and_store():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.search_threads(body={"metadata": {"team": "support"}})
    client.count_threads()
    client.delete_thread("thread-1")
    client.thread_history("thread-1")
    client.resolve_interrupt("thread-1")
    client.get_assistant("assistant-1")
    client.delete_assistant("assistant-1")
    client.search_assistants()
    client.set_latest_assistant_version("assistant-1", 2)
    client.get_run("thread-1", "run-1")
    client.cancel_run("thread-1", "run-1")
    client.delete_run("thread-1", "run-1")
    client.search_runs()
    client.store_put(namespace=["support"], key="customer-1", value={"tier": "gold"})
    client.store_get(namespace=["support"], key="customer-1")
    client.store_search(namespace_prefix=["support"])
    client.store_delete(namespace=["support"], key="customer-1")

    assert calls[0] == (
        "POST",
        "https://example.test/threads/search",
        b'{"metadata": {"team": "support"}}',
    )
    assert calls[-1][0:2] == (
        "DELETE",
        "https://example.test/store/items?namespace=support&key=customer-1",
    )


def test_langsmith_agent_helpers_support_fleet_and_managed_deep_agents():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.list_agents(query={"page_size": 10, "audience": "tenant"})
    client.get_agent("fleet-1", include_files=True)
    client.create_agent(body={"name": "refunds"})
    client.update_agent("fleet-1", body={"description": "Refund assistant"})
    client.delete_agent("fleet-1")
    client.list_agents(path_prefix="/v1/deepagents")

    assert calls[0][1].endswith("/v1/fleet/agents?page_size=10&audience=tenant")
    assert calls[1][1].endswith("/v1/fleet/agents/fleet-1?include_files=true")
    assert calls[2][0:2] == ("POST", "https://example.test/v1/fleet/agents")
    assert calls[5][1] == "https://example.test/v1/deepagents/agents"


def test_langsmith_cursor_iterators_preserve_fleet_filters():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        if "/agents" in url:
            if "cursor=agent-next" in url:
                payload = b'{"items":[{"id":"agent-2"}],"next_cursor":null}'
            else:
                payload = b'{"items":[{"id":"agent-1"}],"next_cursor":"agent-next"}'
        else:
            if "cursor=thread-next" in url:
                payload = b'{"items":[{"id":"thread-2"}],"next_cursor":null}'
            else:
                payload = b'{"items":[{"id":"thread-1"}],"next_cursor":"thread-next"}'
        return 200, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    assert [item["id"] for item in client.iter_agents(query={"name": "refund"})] == [
        "agent-1",
        "agent-2",
    ]
    assert [item["id"] for item in client.iter_fleet_threads(query={"page_size": 10})] == [
        "thread-1",
        "thread-2",
    ]
    client.list_trigger_templates()
    assert "name=refund" in calls[0]
    assert "cursor=agent-next" in calls[1]
    assert "page_size=10" in calls[2]
    assert "cursor=thread-next" in calls[3]
    assert calls[4].endswith("/v1/fleet/trigger-templates")


def test_langsmith_feedback_fleet_thread_and_mcp_helpers_preserve_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.create_feedback(
        run_id="run-1",
        key="user-rating",
        score=1,
        comment="Helpful",
        source_info={"origin": "app"},
    )
    client.list_feedback(query={"run_id": "run-1"})
    client.get_feedback("feedback-1")
    client.update_feedback("feedback-1", body={"comment": "Updated"})
    client.delete_feedback("feedback-1")
    client.list_fleet_threads(query={"page_size": 10})
    client.get_fleet_thread("thread-1")
    client.update_fleet_thread("thread-1", body={"title": "Refunds"})
    client.delete_fleet_thread("thread-1")
    client.list_mcp_servers()
    client.create_mcp_server(body={"name": "search", "url": "https://mcp.example.test"})
    client.get_mcp_server("mcp-1")
    client.update_mcp_server("mcp-1", body={"name": "search-v2"})
    client.delete_mcp_server("mcp-1")

    assert calls[0] == (
        "POST",
        "https://example.test/api/v1/feedback",
        b'{"run_id": "run-1", "key": "user-rating", "score": 1, "comment": "Helpful", "source_info": {"origin": "app"}}',
    )
    assert calls[1][1].endswith("/api/v1/feedback?run_id=run-1")
    assert calls[5][1].endswith("/v1/fleet/threads?page_size=10")
    assert calls[9][1] == "https://example.test/v1/deepagents/mcp-servers"


def test_langsmith_control_plane_client_covers_deployment_and_revision_lifecycle():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithControlPlaneClient(
        api_key="secret",
        tenant_id="workspace-1",
        base_url="https://control.example.test",
        transport=transport,
    )
    client.list_deployments(name_contains="refund")
    client.create_deployment(body={"name": "refunds"})
    client.get_deployment("deployment-1")
    client.patch_deployment("deployment-1", body={"source_config": {"build_on_push": True}})
    client.list_revisions("deployment-1")
    client.get_revision("deployment-1", "revision-1")
    client.create_deployment_revision("deployment-1", body={"config": {}})
    client.get_free_deployment_count()
    assert client.delete_deployment("deployment-1") is None

    assert calls[0][0:2] == (
        "GET",
        "https://control.example.test/v2/deployments?name_contains=refund",
    )
    assert all(call[2]["X-Api-Key"] == "secret" for call in calls)
    assert all(call[2]["X-Tenant-Id"] == "workspace-1" for call in calls)
    assert calls[1][0:2] == ("POST", "https://control.example.test/v2/deployments")
    assert calls[6][1].endswith("/v2/deployments/deployment-1/revisions")


def test_langsmith_control_plane_requires_workspace_identity():
    try:
        LangSmithControlPlaneClient(api_key="secret", base_url="https://example.test")
    except ValueError as exc:
        assert "LANGSMITH_TENANT_ID" in str(exc)
    else:
        raise AssertionError("control-plane client should require a tenant id")
