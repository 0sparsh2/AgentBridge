import asyncio

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

    async def astream_events(self, method, path, *, body=None, **kwargs):
        del method, path, kwargs
        assert body == {"assistant_id": "agent", "input": {"x": 1}}
        yield {"event": "messages", "data": {"content": "hello"}}
        yield {"event": "end", "data": {"output": "hello"}}

    def update_thread_state(self, thread_id, *, values, as_node=None):
        return {"thread_id": thread_id, "values": values, "as_node": as_node}

    def create_thread_run(self, thread_id, *, assistant_id, input, stream=False):
        return {"thread_id": thread_id, "assistant_id": assistant_id, "input": input, "stream": stream}


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
    assert resumed["input"] == {"command": {"resume": "approved"}}


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
    assert resumed["input"] == {"command": {"resume": "approved"}}
