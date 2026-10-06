# Remote Graph Streaming

Use `RemoteGraphClient` to consume a deployed LangGraph or Agent Server graph through AgentBridge
events. Install the `agentbridge-langchain` plugin and set `LANGSMITH_API_KEY` in your environment.
The API client's `base_url` must point to your Agent Server deployment.
Run-option and reconnect request shapes were reviewed against the installed `langgraph-sdk==0.4.4`
implementation. These helpers do not require that SDK package at runtime. The tests verify request
serialization and local streaming behavior; hosted checkpoint replay and event-retention behavior
still require the credentialed deployment lane.

## Receive Events As They Arrive

```python
from contextlib import closing
from agentbridge_langchain import LangSmithAPIClient, RemoteGraphClient

remote = RemoteGraphClient(LangSmithAPIClient(base_url="https://your-agent-server"))
with closing(remote.stream(
    thread_id="existing-thread-id",
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Check refund eligibility"}]},
)) as events:
    for event in events:
        print(event.type, event.data, flush=True)
        print(event.metadata.get("sse_id"))
```

The default HTTP transport yields each complete SSE frame without waiting for the response to
finish. Native event names and IDs are retained as `native_event` and `sse_id` in event metadata.
Namespaced message events retain their native namespace. Multiline JSON data is decoded as a
single event.

Use `closing()` when you may exit the loop early. Closing the iterator releases the HTTP response.
Disconnecting a stream does not cancel the server run; use `remote.cancel_run(thread_id=...,
run_id=...)` to request cancellation separately.

## Async Applications

```python
from contextlib import aclosing

async def receive(remote, thread_id):
    async with aclosing(remote.astream(
        thread_id=thread_id,
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Check my order"}]},
    )) as events:
        async for event in events:
            print(event.type, event.data, flush=True)
```

The async facade reads one frame at a time in a worker thread, keeping blocking HTTP reads off the
application event loop. It does not collect the complete response before yielding. Cancellation
waits for the active blocking read to finish before closing its generator; the default network
read timeout is 30 seconds. It is not an immediate socket-abort API.

## Results And Approvals

`remote.run()` and `remote.arun()` intentionally collect the stream into a `RunResult`. Empty
`end` frames preserve the last state snapshot. Remote error events set `metadata.status` to
`error` without adding a synthetic successful completion event.

Resume an interrupted graph with `remote.resume(..., resume_value=decision)` or `aresume()`.
The decision is sent in the top-level run `command` field. Its shape follows the graph's
interrupt schema. For streaming resumes, use `api.create_thread_run(..., input=None,
command={"resume": decision}, stream=True)` or its async counterpart.

## Native Run Options And Checkpoint Replay

`stream()`, `astream()`, `run()`, and `arun()` accept `run_options`. These fields go into the
native Agent Server request without discarding provider or server-specific settings:

```python
result = remote.run(
    thread_id="existing-thread-id",
    assistant_id="agent",
    input=None,
    run_options={
        "checkpoint": {"checkpoint_id": "saved-checkpoint-id", "checkpoint_ns": ""},
        "stream_mode": ["values", "messages"],
        "stream_resumable": True,
        "durability": "sync",
        "context": {"customer_id": "123"},
        "metadata": {"project": "refunds"},
        "interrupt_before": ["issue_refund"],
    },
)
```

Using `input=None` with a saved checkpoint replays from that snapshot. Execution may repeat
downstream model/tool calls; the server and graph determine persistence and side effects. Use
`command={"resume": decision}` for an interrupt response instead. Supply `assistant_id`, `input`,
and `command` directly, since `run_options` rejects overriding those fields. Other options such
as `config`, `on_disconnect`, `on_completion`, `webhook`, and `multitask_strategy` are preserved.
The server validates them against its version and configuration.

## Reconnect An Existing Run

Start a run with `run_options={"stream_resumable": True}`. Keep the native `run_id` from the
metadata event and the latest `sse_id` from received event metadata. Reconnect explicitly:

```python
with closing(remote.reconnect(
    thread_id="existing-thread-id",
    run_id="saved-run-id",
    last_event_id="saved-sse-id",
    stream_mode=["values", "messages"],
)) as events:
    for event in events:
        print(event.type, event.data)
```

`areconnect()` provides the matching async iterator. Reconnecting joins the existing run at
`/threads/{thread_id}/runs/{run_id}/stream`; it does not submit a new run. It sends `Last-Event-ID`
and preserves requested stream modes. Modes must be a subset of the original run's modes.
`cancel_on_disconnect` defaults to `False`; set it explicitly if disconnecting should cancel the
run. Server retention and resumability settings determine whether old events can be replayed.
Automatic reconnect/retry policies remain application-owned.

## Run The Example

The wire-format demo needs no keys or network access:

```bash
uv run python examples/remote_graph_stream.py
uv run python examples/remote_graph_stream.py --async --max-events 2
uv run python examples/remote_graph_stream.py --resumable --stream-mode values messages
uv run python examples/remote_graph_stream.py --run-id demo --last-event-id demo-1 --async
```

To execute a deployed graph, supply its URL, assistant, and an existing thread:

```bash
uv run python examples/remote_graph_stream.py \
  --url https://your-agent-server --assistant agent --thread existing-thread-id --async
```

An injected legacy `transport=` that returns response bytes remains supported and buffered.
Default HTTP transports stream incrementally. Both LangSmith and Langfuse API clients use the
same incremental parser and async facade; Langfuse SSE methods apply to endpoints that actually
serve SSE, rather than making every JSON endpoint a stream.

## Verification

The local HTTP conformance tests hold the connection open after sending the first event and
verify that the consumer receives it before the server finishes. Additional tests cover early
close, malformed JSON cleanup, async cancellation, namespaced events, and data-only compatibility.

```bash
uv run pytest -q plugins/agentbridge-langchain/tests/test_sse.py \
  plugins/agentbridge-langchain/tests/test_remote_graph.py
```
