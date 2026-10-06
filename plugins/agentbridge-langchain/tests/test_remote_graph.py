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

    def get_thread(self, thread_id):
        return {"thread_id": thread_id, "status": "idle"}

    def get_thread_history(self, thread_id, *, limit=None):
        return {"thread_id": thread_id, "limit": limit, "messages": []}

    def copy_thread(self, thread_id):
        return {"thread_id": thread_id, "copy": True}

    def prune_threads(self, *, thread_ids, strategy):
        return {"thread_ids": thread_ids, "strategy": strategy}

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

    def create_thread_run(self, thread_id, *, assistant_id, input, stream=False):
        return {"thread_id": thread_id, "assistant_id": assistant_id, "input": input, "stream": stream}

    async def aget_thread(self, thread_id):
        return self.get_thread(thread_id)

    async def aget_thread_history(self, thread_id, *, limit=None):
        return self.get_thread_history(thread_id, limit=limit)

    async def acopy_thread(self, thread_id):
        return self.copy_thread(thread_id)

    async def aprune_threads(self, *, thread_ids, strategy):
        return self.prune_threads(thread_ids=thread_ids, strategy=strategy)

    async def aget_thread_state_at_checkpoint(self, thread_id, checkpoint_id, *, subgraphs=None):
        return self.get_thread_state_at_checkpoint(thread_id, checkpoint_id, subgraphs=subgraphs)

    async def aget_run(self, thread_id, run_id):
        return self.get_run(thread_id, run_id)

    async def alist_run_events(self, thread_id, run_id):
        return self.list_run_events(thread_id, run_id)

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
    assert resumed["input"] == {"command": {"resume": "approved"}}


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
