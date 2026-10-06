# Remote Graph Streaming

Use `RemoteGraphClient` to consume a deployed LangGraph or Agent Server graph through AgentBridge
events. Install the `agentbridge-langchain` plugin and set `LANGSMITH_API_KEY` in your environment.
The API client's `base_url` must point to your Agent Server deployment.

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

## Run The Example

The wire-format demo needs no keys or network access:

```bash
uv run python examples/remote_graph_stream.py
uv run python examples/remote_graph_stream.py --async --max-events 2
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
