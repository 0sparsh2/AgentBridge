import asyncio
import json

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
