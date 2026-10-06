import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

from agentbridge_langchain.langsmith_api import LangSmithAPIClient
from agentbridge_langchain.remote_graph import RemoteGraphClient


class FakeClient:
    def create_thread(self, *, metadata=None):
        return {"thread_id": "thread-1", "metadata": metadata}

    def stream_thread_run(self, thread_id, *, assistant_id, input):
        assert (thread_id, assistant_id, input) == ("thread-1", "agent", {"x": 1})
        yield {"event": "metadata", "data": {"run_id": "run-1"}}
        yield {"event": "messages", "data": {"content": "hello"}}
        yield {"event": "end", "data": {"output": "hello"}}

    def get_thread_state(self, thread_id, *, checkpoint_id=None):
        return {"thread_id": thread_id, "checkpoint_id": checkpoint_id}

    def get_thread(self, thread_id):
        return {"thread_id": thread_id, "status": "idle"}

    def get_thread_history(self, thread_id, *, limit=None):
        return {"thread_id": thread_id, "limit": limit, "messages": []}

    def copy_thread(self, thread_id):
        return {"thread_id": thread_id, "copy": True}

    def prune_threads(self, *, thread_ids, strategy):
        return {"thread_ids": thread_ids, "strategy": strategy}

    def search_threads(self, *, body=None):
        return {"body": body, "threads": []}

    def resolve_interrupt(self, thread_id):
        return {"thread_id": thread_id, "resolved": True}

    def get_assistant(self, assistant_id):
        return {"assistant_id": assistant_id, "graph_id": "refund"}

    def get_assistant_graph(self, assistant_id, *, xray=None):
        return {"assistant_id": assistant_id, "xray": xray, "nodes": []}

    def get_assistant_schemas(self, assistant_id):
        return {"assistant_id": assistant_id, "input_schema": {}}

    def get_assistant_subgraphs(self, assistant_id, *, namespace=None):
        return {"assistant_id": assistant_id, "namespace": namespace, "subgraphs": []}

    def get_assistant_versions(self, assistant_id, *, query=None):
        return {"assistant_id": assistant_id, "query": query, "versions": []}

    def set_latest_assistant_version(self, assistant_id, version):
        return {"assistant_id": assistant_id, "version": version}

    def get_thread_state_at_checkpoint(self, thread_id, checkpoint_id, *, subgraphs=None):
        return {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "subgraphs": subgraphs,
        }

    def get_run(self, thread_id, run_id):
        return {"thread_id": thread_id, "run_id": run_id, "status": "success"}

    def list_run_events(self, thread_id, run_id):
        return [{"thread_id": thread_id, "run_id": run_id, "event": "done"}]

    def list_thread_runs(self, thread_id, *, query=None):
        return {"thread_id": thread_id, "query": query, "runs": []}

    def join_run(self, thread_id, run_id):
        return {"thread_id": thread_id, "run_id": run_id, "output": "hello"}

    def cancel_run(self, thread_id, run_id):
        return {"thread_id": thread_id, "run_id": run_id, "cancelled": True}

    def delete_run(self, thread_id, run_id):
        return {"thread_id": thread_id, "run_id": run_id, "deleted": True}

    async def astream_events(self, method, path, *, body=None, **kwargs):
        del method, path, kwargs
        assert body == {"assistant_id": "agent", "input": {"x": 1}}
        yield {"event": "messages", "data": {"content": "hello"}}
        yield {"event": "end", "data": {"output": "hello"}}

    def update_thread_state(self, thread_id, *, values, as_node=None):
        return {"thread_id": thread_id, "values": values, "as_node": as_node}

    def create_thread_run(self, thread_id, *, assistant_id, input, stream=False, command=None):
        return {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "input": input,
            "stream": stream,
            "command": command,
        }

    async def aget_thread(self, thread_id):
        return self.get_thread(thread_id)

    async def aget_thread_history(self, thread_id, *, limit=None):
        return self.get_thread_history(thread_id, limit=limit)

    async def acopy_thread(self, thread_id):
        return self.copy_thread(thread_id)

    async def aprune_threads(self, *, thread_ids, strategy):
        return self.prune_threads(thread_ids=thread_ids, strategy=strategy)

    async def asearch_threads(self, *, body=None):
        return self.search_threads(body=body)

    async def aresolve_interrupt(self, thread_id):
        return self.resolve_interrupt(thread_id)

    async def aget_assistant(self, assistant_id):
        return self.get_assistant(assistant_id)

    async def aget_assistant_graph(self, assistant_id, *, xray=None):
        return self.get_assistant_graph(assistant_id, xray=xray)

    async def aget_assistant_schemas(self, assistant_id):
        return self.get_assistant_schemas(assistant_id)

    async def aget_assistant_subgraphs(self, assistant_id, *, namespace=None):
        return self.get_assistant_subgraphs(assistant_id, namespace=namespace)

    async def aget_assistant_versions(self, assistant_id, *, query=None):
        return self.get_assistant_versions(assistant_id, query=query)

    async def aset_latest_assistant_version(self, assistant_id, version):
        return self.set_latest_assistant_version(assistant_id, version)

    async def aget_thread_state_at_checkpoint(self, thread_id, checkpoint_id, *, subgraphs=None):
        return self.get_thread_state_at_checkpoint(thread_id, checkpoint_id, subgraphs=subgraphs)

    async def aget_run(self, thread_id, run_id):
        return self.get_run(thread_id, run_id)

    async def alist_run_events(self, thread_id, run_id):
        return self.list_run_events(thread_id, run_id)

    async def alist_thread_runs(self, thread_id, *, query=None):
        return self.list_thread_runs(thread_id, query=query)

    async def ajoin_run(self, thread_id, run_id):
        return self.join_run(thread_id, run_id)

    async def acancel_run(self, thread_id, run_id):
        return self.cancel_run(thread_id, run_id)

    async def adelete_run(self, thread_id, run_id):
        return self.delete_run(thread_id, run_id)


def test_remote_graph_normalizes_sse_lifecycle_and_result():
    client = RemoteGraphClient(FakeClient())
    assert client.create_thread(metadata={"team": "support"}) == "thread-1"
    result = client.run(thread_id="thread-1", assistant_id="agent", input={"x": 1})
    assert result.backend == "langgraph_remote"
    assert result.output == "hello"
    assert [event.type for event in result.events] == ["workflow", "message", "complete"]


def test_remote_graph_state_and_resume_helpers_preserve_native_payloads():
    client = RemoteGraphClient(FakeClient())
    assert client.state("thread-1", checkpoint_id="cp-1")["checkpoint_id"] == "cp-1"
    assert client.update_state("thread-1", values={"approved": True}, as_node="review")["as_node"] == "review"
    resumed = client.resume(thread_id="thread-1", assistant_id="agent", resume_value="approved")
    assert resumed["input"] is None
    assert resumed["command"] == {"resume": "approved"}


def test_remote_graph_lifecycle_helpers_preserve_native_thread_and_run_payloads():
    client = RemoteGraphClient(FakeClient())

    assert client.thread("thread-1")["status"] == "idle"
    assert client.history("thread-1", limit=4)["limit"] == 4
    assert client.copy_thread("thread-1")["copy"] is True
    assert client.prune_threads(thread_ids=["thread-1"], strategy="delete") == {
        "thread_ids": ["thread-1"],
        "strategy": "delete",
    }
    assert client.state_at_checkpoint("thread-1", "cp-1", subgraphs=True)["subgraphs"] is True
    assert client.run_record(thread_id="thread-1", run_id="run-1")["status"] == "success"
    assert client.run_events(thread_id="thread-1", run_id="run-1")[0]["event"] == "done"
    assert client.join_run(thread_id="thread-1", run_id="run-1")["output"] == "hello"
    assert client.cancel_run(thread_id="thread-1", run_id="run-1")["cancelled"] is True
    assert client.delete_run(thread_id="thread-1", run_id="run-1")["deleted"] is True


def test_remote_graph_discovery_helpers_preserve_native_filters_and_versions():
    client = RemoteGraphClient(FakeClient())

    assert client.search_threads(body={"metadata": {"team": "support"}})["threads"] == []
    assert client.resolve_interrupt("thread-1")["resolved"] is True
    assert client.assistant("agent")["graph_id"] == "refund"
    assert client.assistant_graph("agent", xray=True)["xray"] is True
    assert client.assistant_schemas("agent")["input_schema"] == {}
    assert client.assistant_subgraphs("agent", namespace="review")["namespace"] == "review"
    assert client.assistant_versions("agent", query={"limit": 2})["query"] == {"limit": 2}
    assert client.set_latest_assistant_version("agent", 3)["version"] == 3
    assert client.thread_runs("thread-1", query={"limit": 4})["query"] == {"limit": 4}


def test_remote_graph_supports_async_native_event_stream():
    client = RemoteGraphClient(FakeClient())

    async def collect():
        return await client.arun(thread_id="thread-1", assistant_id="agent", input={"x": 1})

    result = asyncio.run(collect())
    assert result.output == "hello"
    assert result.metadata["async"] is True
    assert result.events[-1].type == "complete"


def test_remote_graph_async_thread_state_and_resume_helpers_use_compatibility_fallbacks():
    client = RemoteGraphClient(FakeClient())

    async def collect():
        thread_id = await client.acreate_thread(metadata={"team": "support"})
        state = await client.astate(thread_id, checkpoint_id="cp-2")
        updated = await client.aupdate_state(thread_id, values={"approved": True}, as_node="review")
        resumed = await client.aresume(
            thread_id=thread_id,
            assistant_id="agent",
            resume_value="approved",
        )
        return thread_id, state, updated, resumed

    thread_id, state, updated, resumed = asyncio.run(collect())
    assert thread_id == "thread-1"
    assert state["checkpoint_id"] == "cp-2"
    assert updated["as_node"] == "review"
    assert resumed["input"] is None
    assert resumed["command"] == {"resume": "approved"}


def test_remote_graph_async_lifecycle_helpers_preserve_native_payloads():
    client = RemoteGraphClient(FakeClient())

    async def collect():
        return (
            await client.athread("thread-1"),
            await client.ahistory("thread-1", limit=2),
            await client.acopy_thread("thread-1"),
            await client.aprune_threads(thread_ids=["thread-1"], strategy="delete"),
            await client.astate_at_checkpoint("thread-1", "cp-1", subgraphs=False),
            await client.arun_record(thread_id="thread-1", run_id="run-1"),
            await client.arun_events(thread_id="thread-1", run_id="run-1"),
            await client.ajoin_run(thread_id="thread-1", run_id="run-1"),
            await client.acancel_run(thread_id="thread-1", run_id="run-1"),
            await client.adelete_run(thread_id="thread-1", run_id="run-1"),
        )

    values = asyncio.run(collect())
    assert values[0]["thread_id"] == "thread-1"
    assert values[1]["limit"] == 2
    assert values[2]["copy"] is True
    assert values[3]["strategy"] == "delete"
    assert values[4]["checkpoint_id"] == "cp-1"
    assert values[5]["run_id"] == "run-1"
    assert values[6][0]["event"] == "done"
    assert values[7]["output"] == "hello"
    assert values[8]["cancelled"] is True
    assert values[9]["deleted"] is True


def test_remote_graph_async_discovery_helpers_preserve_native_payloads():
    client = RemoteGraphClient(FakeClient())

    async def collect():
        return (
            await client.asearch_threads(body={"limit": 1}),
            await client.aresolve_interrupt("thread-1"),
            await client.aassistant("agent"),
            await client.aassistant_graph("agent", xray=2),
            await client.aassistant_schemas("agent"),
            await client.aassistant_subgraphs("agent", namespace="review"),
            await client.aassistant_versions("agent", query={"limit": 2}),
            await client.aset_latest_assistant_version("agent", 3),
            await client.athread_runs("thread-1", query={"limit": 4}),
        )

    values = asyncio.run(collect())
    assert values[0]["body"] == {"limit": 1}
    assert values[1]["resolved"] is True
    assert values[2]["assistant_id"] == "agent"
    assert values[3]["xray"] == 2
    assert values[4]["input_schema"] == {}
    assert values[5]["namespace"] == "review"
    assert values[6]["query"] == {"limit": 2}
    assert values[7]["version"] == 3
    assert values[8]["query"] == {"limit": 4}


def test_remote_resume_serializes_top_level_command_through_real_api_client():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, json.loads(body) if body else None))
        return 200, {"content-type": "application/json"}, b'{"run_id":"run-1"}'

    api = LangSmithAPIClient(
        api_key="test-key", base_url="https://server.test", transport=transport
    )
    remote = RemoteGraphClient(api)
    decision = {"interrupt-1": {"approved": True}}
    remote.resume(thread_id="thread-1", assistant_id="agent", resume_value=decision)

    async def collect():
        await remote.aresume(
            thread_id="thread-1", assistant_id="agent", resume_value=decision
        )
        await remote.adelete_run(thread_id="thread-1", run_id="run-1")

    asyncio.run(collect())
    expected = (
        "POST",
        "https://server.test/threads/thread-1/runs",
        {"assistant_id": "agent", "input": None, "command": {"resume": decision}},
    )
    assert calls[:2] == [expected, expected]
    assert calls[2] == ("DELETE", "https://server.test/threads/thread-1/runs/run-1", None)


def test_thread_run_stream_preserves_resume_command_for_sync_and_async_requests():
    calls = []

    def transport(method, url, headers, body):
        calls.append(json.loads(body))
        return 200, {"content-type": "text/event-stream"}, b'data: {"output":"approved"}\n\n'

    api = LangSmithAPIClient(api_key="test-key", transport=transport)
    assert list(api.create_thread_run(
        "thread-1", assistant_id="agent", input=None, stream=True, command={"resume": False}
    )) == [{"output": "approved"}]

    async def collect():
        stream = await api.acreate_thread_run(
            "thread-1", assistant_id="agent", input=None, stream=True, command={"resume": False}
        )
        return [event async for event in stream]

    assert asyncio.run(collect()) == [{"output": "approved"}]
    assert calls == [
        {"assistant_id": "agent", "input": None, "command": {"resume": False}},
        {"assistant_id": "agent", "input": None, "command": {"resume": False}},
    ]


def test_remote_graph_normalizes_wire_event_names_and_resume_ids():
    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: metadata\nid: run-1-0\ndata: {"run_id":"run-1"}\n\n'
            b'event: messages-tuple|review\nid: run-1-1\n'
            b'data: [{"content":"hello"},{"node":"review"}]\n\n'
            b'event: error\nid: run-1-2\ndata: {"message":"failed"}\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    events = list(remote.stream(thread_id="thread-1", assistant_id="agent", input={}))

    async def collect():
        return [event async for event in remote.astream(
            thread_id="thread-1", assistant_id="agent", input={}
        )]

    async_events = asyncio.run(collect())
    assert [event.type for event in events] == ["workflow", "message", "error"]
    assert [event.model_dump() for event in async_events] == [
        event.model_dump() for event in events
    ]
    assert events[1].data["value"][0]["content"] == "hello"
    assert events[1].metadata == {
        "native_event": "messages-tuple|review", "sse_id": "run-1-1"
    }
    assert events[2].metadata["sse_id"] == "run-1-2"


def test_remote_run_preserves_final_state_when_end_frame_has_no_payload():
    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: values\ndata: {"answer":"approved"}\n\n'
            b'event: end\ndata: null\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    result = remote.run(thread_id="thread-1", assistant_id="agent", input={})
    assert result.output == {"answer": "approved"}
    assert result.metadata["status"] == "completed"


def test_remote_error_does_not_synthesize_success_completion():
    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: error\ndata: {"message":"model failed"}\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    result = remote.run(thread_id="thread-1", assistant_id="agent", input={})

    async def collect():
        return await remote.arun(thread_id="thread-1", assistant_id="agent", input={})

    async_result = asyncio.run(collect())
    for value in (result, async_result):
        assert value.output is None
        assert value.metadata["status"] == "error"
        assert [event.type for event in value.events] == ["error"]


def test_remote_stream_close_releases_sync_and_async_native_streams():
    closed = []

    class Client:
        def stream_thread_run(self, *args, **kwargs):
            try:
                yield {"event": "messages", "data": {"content": "first"}}
                raise AssertionError("read ahead")
            finally:
                closed.append("sync")

        async def astream_events(self, *args, **kwargs):
            try:
                yield {"event": "messages", "data": {"content": "first"}}
                raise AssertionError("read ahead")
            finally:
                closed.append("async")

    remote = RemoteGraphClient(Client())
    events = remote.stream(thread_id="thread-1", assistant_id="agent", input={})
    assert next(events).data["content"] == "first"
    events.close()
    assert closed == ["sync"]

    async def collect():
        events = remote.astream(thread_id="thread-1", assistant_id="agent", input={})
        assert (await anext(events)).data["content"] == "first"
        await events.aclose()
        assert closed == ["sync", "async"]

    asyncio.run(collect())


def test_remote_run_options_and_checkpoint_replay_preserve_sync_async_request_bodies():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, json.loads(body)))
        return 200, {"content-type": "text/event-stream"}, (
            b'event: values\ndata: {"eligible":true}\n\nevent: end\ndata: null\n\n'
        )

    api = LangSmithAPIClient(api_key="test", transport=transport)
    remote = RemoteGraphClient(api)
    options = {
        "stream_mode": ["values", "messages"], "stream_resumable": True,
        "checkpoint": {"checkpoint_id": "cp-1", "checkpoint_ns": "review"},
        "context": {"customer": "123"}, "metadata": {"project": "refunds"},
        "config": {"configurable": {"policy": "refund"}}, "durability": "sync",
        "interrupt_before": ["issue_refund"], "on_disconnect": "continue",
    }
    decision = {"resume": {"approved": True}}
    result = remote.run(
        thread_id="thread-1", assistant_id="agent", input=None,
        command=decision, run_options=options,
    )

    async def collect():
        return await remote.arun(
            thread_id="thread-1", assistant_id="agent", input=None,
            command=decision, run_options=options,
        )

    assert result.output == asyncio.run(collect()).output == {"eligible": True}
    expected = {"assistant_id": "agent", "input": None, "command": decision, **options}
    assert [call[2] for call in calls] == [expected, expected]
    assert "assistant_id" not in options


@pytest.mark.parametrize("reserved", ["assistant_id", "input", "command"])
def test_remote_run_options_reject_reserved_overrides_before_transport(reserved):
    def transport(*args):
        raise AssertionError("invalid request reached transport")

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    with pytest.raises(ValueError, match="reserved fields"):
        remote.run(thread_id="thread-1", assistant_id="agent", input={},
                   run_options={reserved: "override"})

    async def collect():
        return await remote.arun(thread_id="thread-1", assistant_id="agent", input={},
                                 run_options={reserved: "override"})

    with pytest.raises(ValueError, match="reserved fields"):
        asyncio.run(collect())


def test_remote_reconnect_uses_existing_run_stream_with_last_event_id():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        return 200, {"content-type": "text/event-stream"}, (
            b'event: messages\nid: run-1-3\ndata: {"content":"resumed stream"}\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    args = {
        "thread_id": "thread-1", "run_id": "run-1", "last_event_id": "run-1-2",
        "stream_mode": ["values", "messages"], "cancel_on_disconnect": True,
    }
    events = list(remote.reconnect(**args))

    async def collect():
        return [event async for event in remote.areconnect(**args)]

    assert events[0].model_dump() == asyncio.run(collect())[0].model_dump()
    assert events[0].metadata["sse_id"] == "run-1-3"
    for method, url, headers, body in calls:
        parsed = urlparse(url)
        assert method == "GET"
        assert parsed.path == "/threads/thread-1/runs/run-1/stream"
        assert parse_qs(parsed.query) == {
            "stream_mode": ["values", "messages"], "cancel_on_disconnect": ["True"]
        }
        assert headers["Last-Event-ID"] == "run-1-2"
        assert body is None


def test_async_sync_only_client_streams_without_collecting_later_frames():
    closed = []

    class Client:
        def stream_thread_run(self, thread_id, *, assistant_id, input, run_options=None):
            assert run_options == {"stream_resumable": True}
            try:
                yield {"event": "messages", "data": {"content": "first"}}
                raise AssertionError("sync compatibility path collected the rest of the stream")
            finally:
                closed.append(True)

    async def collect():
        events = RemoteGraphClient(Client()).astream(
            thread_id="thread-1", assistant_id="agent", input={},
            run_options={"stream_resumable": True},
        )
        assert (await anext(events)).data["content"] == "first"
        await events.aclose()
        assert closed == [True]

    asyncio.run(collect())


@pytest.mark.parametrize("event_name", ["updates", "values", "updates|review"])
def test_remote_interrupt_preserves_payload_and_never_reports_success(event_name):
    interrupt = {
        "id": "approval-1", "value": {"action": "refund", "amount": 25},
        "resumable": True, "ns": ["review:task-1"],
    }
    state = {"order_id": "A123", "__interrupt__": [interrupt]}

    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: metadata\ndata: {"run_id":"run-1"}\n\n'
            + f'event: {event_name}\nid: run-1-2\ndata: {json.dumps(state)}\n\n'.encode()
            + b'event: end\ndata: null\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    result = remote.run(thread_id="thread-1", assistant_id="agent", input={})

    async def collect():
        return await remote.arun(thread_id="thread-1", assistant_id="agent", input={})

    async_result = asyncio.run(collect())
    for value in (result, async_result):
        assert value.metadata["status"] == "interrupted"
        assert value.metadata["interrupted"] is True
        assert value.metadata["interrupts"] == [interrupt]
        assert value.metadata["run_id"] == "run-1"
        assert not any(event.type == "complete" for event in value.events)
        paused = value.events[1]
        assert paused.data == {"phase": "interrupted", "interrupts": [interrupt], "native_data": state}
        assert paused.metadata["native_event"] == event_name
        assert paused.metadata["sse_id"] == "run-1-2"
        assert value.events[-1].data["status"] == "interrupted"
        if event_name == "values":
            assert value.output == state


def test_remote_interrupt_can_resume_to_completed_result_without_stale_pause_state():
    calls = []

    def transport(method, url, headers, body):
        request = json.loads(body)
        calls.append(request)
        if "command" in request:
            response = b'event: values\ndata: {"refunded":true}\n\nevent: end\ndata: null\n\n'
        else:
            response = (
                b'event: updates\ndata: {"__interrupt__":[{"id":"approval-1","value":"Approve?"}]}\n\n'
                b'event: end\ndata: null\n\n'
            )
        return 200, {"content-type": "text/event-stream"}, response

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    first = remote.run(thread_id="thread-1", assistant_id="agent", input={})
    second = remote.run(thread_id="thread-1", assistant_id="agent", input=None,
                        command={"resume": {"approval-1": True}})
    assert first.metadata["status"] == "interrupted"
    assert second.metadata["status"] == "completed"
    assert "interrupted" not in second.metadata
    assert second.output == {"refunded": True}
    assert second.events[-1].type == "complete"
    assert calls[-1]["command"] == {"resume": {"approval-1": True}}


def test_remote_application_state_fields_do_not_trigger_interrupt_status():
    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: values\ndata: {"phase":"interrupted","value":25,"__interrupt__":[]}\n\n'
            b'event: end\ndata: null\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    result = remote.run(thread_id="thread-1", assistant_id="agent", input={})
    assert result.metadata["status"] == "completed"
    assert result.output == {"phase": "interrupted", "value": 25, "__interrupt__": []}


def test_remote_error_after_interrupt_takes_precedence_and_preserves_diagnostics():
    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: interrupts\ndata: [{"id":"approval-1","value":false}]\n\n'
            b'event: error\ndata: {"message":"checkpoint failed"}\n\n'
            b'event: end\ndata: null\n\n'
        )

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    result = remote.run(thread_id="thread-1", assistant_id="agent", input={})
    assert result.metadata["status"] == "error"
    assert result.metadata["interrupts"] == [{"id": "approval-1", "value": False}]
    assert not any(event.type == "complete" for event in result.events)
    assert result.events[-1].data["status"] == "error"


def test_native_local_graph_approval_demo_preserves_checkpoint_resume_semantics():
    pytest.importorskip("langgraph")
    from examples.remote_graph_approval import run_demo

    report = run_demo()
    assert report["initial"]["metadata"]["status"] == "interrupted"
    assert report["initial"]["metadata"]["interrupts"][0]["value"] == {
        "order_id": "A123", "amount": 25,
    }
    assert report["resumed"]["metadata"]["status"] == "completed"
    assert report["resumed"]["output"] == {"order_id": "A123", "refunded": True}
    assert len(report["requests"]) == 2
    assert report["requests"][1]["input"] is None


def test_native_static_breakpoint_is_interrupted_with_no_approval_payload():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(dict)
    builder.add_node("step", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["step"])

    def transport(method, url, headers, body):
        request = json.loads(body)
        frames = []
        for mode, chunk in graph.stream(
            request["input"], {"configurable": {"thread_id": "static-demo"}},
            stream_mode=["values", "updates"],
        ):
            frames.append(f'event: {mode}\ndata: {json.dumps(chunk)}\n\n'.encode())
        frames.append(b'event: end\ndata: null\n\n')
        return 200, {"content-type": "text/event-stream"}, b"".join(frames)

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="test", transport=transport))
    paused = remote.run(thread_id="static-demo", assistant_id="agent", input={"value": 1})
    assert paused.metadata["status"] == "interrupted"
    assert paused.metadata["interrupts"] == []
    assert paused.output == {"value": 1}
    assert not any(event.type == "complete" for event in paused.events)
    continued = remote.run(thread_id="static-demo", assistant_id="agent", input=None)
    assert continued.metadata["status"] == "completed"
    assert continued.output == {"value": 2}
