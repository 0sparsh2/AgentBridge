# agentbridge-langchain

AgentBridge adapter plugin for `langchain`.

## Adopted Framework Version

- Native package: `langchain`
- Adopted range: `>=1.4,<2`
- Verified locally: `1.4.0`
- Status: partial native adapter
- Optional MCP package: `langchain-mcp-adapters>=0.1,<1`

## Target Capabilities

- Direct LangChain `create_agent` compatibility alongside the built-in LangGraph backend.
- `ToolSpec` to `StructuredTool` mapping.
- Structured output through native LangChain `response_format` and typed `structured_response`.
- Middleware, callbacks, memory hints, retriever hints, and native `create_agent` options through
  `LangChainExtension`.
- Native per-run `RunnableConfig` controls such as `configurable`, `max_concurrency`,
  `recursion_limit`, and `run_id` can be supplied through `runtime_config`; session IDs merge into
  `configurable.thread_id` without discarding caller-provided values.
- Native stream controls can be passed through `stream_options` (for example `stream_mode`,
  `version`, and future LangGraph stream keywords), while `stream_events_version` selects the
  event protocol version for `stream_events`/`astream_events`.
- Explicit native retrievers can be exposed as agent tools with
  `LangChainExtension.config(retriever_tools=[...])`; document content and metadata are preserved
  in tool results while the original retriever remains the implementation of record.
- Runtime context schemas receive `RunInput.context` through LangChain's native `context=`
  invocation channel instead of being mixed into message state.
- Native `HumanInTheLoopMiddleware` pauses are surfaced as normalized workflow interrupt events and
  preserve the native checkpoint state for the caller's approval/resume flow.
- Native MCP adapter tools can be supplied with `LangChainExtension(mcp_tools=[...])` and are
  forwarded unchanged to `create_agent`; `langchain-mcp-adapters` remains optional.
- OpenAI-compatible native models can use `model_provider="openai"` and `model_options={...}`
  for NVIDIA NIM, OpenRouter, or internal gateways; install the optional `openai` extra.
- Local Ollama models can use `model_provider="ollama"` and `model_options={...}`; install the
  optional `ollama` extra. A custom `base_url` can be passed through unchanged.
- Provider-specific factories lazily support Anthropic, Google GenAI/Vertex AI, Mistral, Groq,
  Cohere, Bedrock, Fireworks, Hugging Face, xAI, and Azure OpenAI when their native LangChain
  integration package is installed. Provider packages remain optional; a pre-built native model
  object is always accepted as an escape hatch.
- Any other LangChain provider can pass its already-constructed native chat model through
  `LangChainExtension.with_config(agent, model=native_model)`, keeping provider-specific packages
  and features outside the core install path.
- Native model resilience is available through `model_retry={...}` and
  `model_fallbacks=[native_model, ...]`; AgentBridge applies LangChain `with_retry()` before
  `with_fallbacks()` and rejects ambiguous string fallback entries.
- A guarded `native_options` escape hatch for newly released LangChain `create_agent` options;
  AgentSpec-owned identity/model/tool fields cannot be overridden.
- LangSmith tracing context and Langfuse callback integration through one AgentBridge observability config.
- LangSmith dataset publishing and hosted evaluation through an optional integration module.
- LangSmith dataset and example CRUD plus cursor-safe iteration through the dependency-free API
  client, preserving native inputs, outputs, metadata, and dataset filters.
- LangSmith dataset regression deltas and shared-dataset examples with runs are available through
  typed sync and async API-client helpers, preserving native comparison and sharing payloads.
- LangSmith dataset version reads/diffs, split listing/updates, and public share/unshare lifecycle
  are also available through typed sync and async helpers.
- LangSmith bulk example deletion, public shared-example reads, and OpenAI fine-tuning dataset
  export are available through typed sync and async helpers; attachment-heavy multipart uploads
  remain available through the generic transport escape hatch.
- LangSmith Agent Auth connection create/list/remove operations through the dependency-free API
  client, preserving native connection payloads and agent-scoped paths.
- LangSmith Fleet Agent Auth connection-token list/update/revoke operations are available through
  typed sync and async helpers.
- LangSmith annotation-queue creation, rubric metadata, run assignment, review-status listing,
  indexed retrieval, and removal are available through typed sync and async helpers.
- LangSmith feedback configuration CRUD and presigned browser feedback-token creation/listing are
  available through typed sync and async helpers; deprecated composite feedback formulas remain
  intentionally native-only.
- LangSmith feedback creation now preserves native trace/session correlation, evaluator/source
  metadata, corrections, grouping, comparative experiments, retention, and error fields.
- Async LangSmith governance helpers cover Agent Auth connections, platform tool registry CRUD by
  ID or handle, and long-term store put/get/search/delete operations.
- Async Fleet and Deep Agents helpers cover agent/thread CRUD, cursor iteration, MCP-server
  registration, and trigger-template discovery with configurable native path prefixes.
- LangSmith deployment control-plane revision redeploy/interruption, bulk deletion, deployment and
  revision logs, and resource/deployment-tier updates are exposed through typed helpers.
- LangSmith v2 listener create/list/get/patch/delete operations are exposed through the same
  dependency-free control-plane client.
- LangSmith platform tool-registry create/list/get/update/delete operations are exposed by ID and
  handle, allowing AgentBridge tool definitions to be synchronized without the LangSmith SDK.
- `publish_dataset()` supports both the optional native LangSmith SDK and the dependency-free
  `LangSmithAPIClient` dataset/example transport.
- Async LangSmith data-plane facades cover dataset and example CRUD, cursor iteration, and full
  feedback retrieval/update/deletion alongside the existing feedback creation helper.
- Async LangSmith typed facades cover assistant CRUD/introspection/version selection, thread
  creation/search/history/state/checkpoint updates/copying, thread-run inspection/join/cancellation,
  interrupt resolution, pruning, stateless runs, health/info/docs/metrics, feedback, A2A JSON-RPC, MCP,
  cron scheduling, and the existing SSE stream facade. The control
  plane client also exposes async deployment/revision, logs, tier, and listener lifecycles while
  retaining the workspace tenant header.
- Remote LangGraph/Agent Server thread and run streaming through `RemoteGraphClient`, normalized into
  `AgentEvent` and `RunResult`; thread history/copy/pruning, checkpoint snapshots, and run
  inspection/events/join/cancel/delete, assistant graph/schema/version discovery, thread search,
  and interrupt resolution are available through sync and async lifecycle helpers.
- Langfuse callback wiring plus a dependency-free JSON/SSE/raw API transport for telemetry and export endpoints.
- Langfuse OTLP trace ingestion with the v4 ingestion header, cursor-safe Observations v2 and Scores v3 iteration, Metrics v2, Experiments API, trace, score, dataset, and dataset-item helpers.
- Langfuse current v2 dataset and versioned dataset-item lifecycle operations, including archive/upsert and trace cleanup helpers.
- Async Langfuse facades also cover Metrics v2, prompt listing/numbered-page iteration, legacy trace
  retrieval/deletion, and dataset-item retrieval/deletion with the same native paths.
- Async Langfuse cursor iterators cover experiments and experiment items. Prompts, datasets,
  and dataset items use native numbered-page iterators with server metadata validation; both
  styles preserve caller filters.
- Async Langfuse dataset deletion and dataset-item update/upsert now match the synchronous
  lifecycle surface.
- Langfuse prompt version/label retrieval and text/chat variable compilation through `LangfusePrompt`.
- Async `afetch_prompt()` provides the same version/label retrieval and compilation bridge without
  blocking an async application.
- Langfuse evaluation bridge for publishing `EvaluationExample` datasets and normalized report scores.
- Report publication preserves explicit evaluator values, comments, and metadata, including
  false booleans, zero, and empty text. `score_types` supports categorical strings; dataset
  publication preserves example metadata. Scalar validation runs before score writes begin.
- Sync `iter_scores_v3()` and async `aiter_scores_v3()` preserve typed values, subjects, and
  caller filters, and reject cursor cycles or malformed pages rather than reporting a complete
  export. See `examples/langfuse_typed_evaluation.py` for an offline publish/export round trip.
- The evaluation bridge also exposes async `apublish_dataset()` and `apublish_report_scores()`
  helpers for fully asynchronous evaluation pipelines.
- These async evaluation helpers are exported from the plugin package as
  `apublish_langfuse_dataset()` and `apublish_report_scores()`.
- Langfuse score creation preserves typed numeric, boolean, categorical, and text values plus
  trace/session/observation/dataset-run targets, idempotency IDs, score configs, metadata, and
  environment fields.
- Current Langfuse v4 reads use `list_trace_observations()` and `get_observation()` for bounded
  Observations v2 queries, and `get_score()` for Scores v3 lookup; deprecated trace and
  observation-by-ID reads are not presented as current APIs.
- Cursor-safe `iter_experiments()` and `iter_experiment_items()` helpers support evaluation
  exports and CI regression gates without dropping the original filters between pages.
- Cursor-safe `iter_dataset_items()` supports versioned dataset exports without dropping dataset
  or version filters between pages.
- Numbered-page `iter_datasets()` and `iter_prompts()` support complete Langfuse evaluation and
  prompt-management exports without dropping caller filters.
- Async Langfuse query facades cover observations, scores, experiments, datasets, and dataset items;
  `aiter_observations()` preserves the current cursor contract.
- Async Langfuse lifecycle facades cover prompt versions, typed scores, datasets, and dataset-item
  upserts using the same current v2/v3 paths as synchronous helpers.
- Streaming normalization for LangChain `stream_events(..., version="v3")` event envelopes and
  `stream(..., stream_mode=["messages", "updates", "custom"], version="v2")` chunks, with
  `stream_options`, `stream_events_version`, and `stream_protocol="auto|events|stream"` controls
  for selecting the native transport explicitly.
- Async normalized execution and streaming through `arun_agent()` and `astream_agent()`. When the
  native runtime exposes `ainvoke`, `astream`, or `astream_events`, the adapter uses those methods;
  otherwise it retains the dependency-free fallback contract.
- Ordered batch and async-batch execution through `batch_agent()` and `abatch_agent()`. The adapter
  uses LangChain `batch()`/`abatch()` when available and falls back to normalized sequential runs.

## Install

```bash
pip install -e .
```

## Verify Discovery

```bash
agentbridge plugins
agentbridge list-backends
agentbridge inspect-backend langchain --json
agentbridge conformance --backend langchain
```

The conformance runner uses the plugin-only `agentbridge/offline` model string. That path still
builds a LangChain `create_agent` graph and executes it natively, but uses a tiny local
`BaseChatModel` implementation so contract checks do not require provider packages or API keys.

## LangChain vs LangGraph

Use `langchain` when you are migrating or standardizing an existing LangChain agent app that depends
on `create_agent`, `StructuredTool`, middleware, callbacks, memory, retrievers, or LangSmith-style
instrumentation.

Use `langgraph` when the app is primarily a durable workflow: explicit graph topology, checkpoints,
interrupts, resumability, state routing, or production orchestration are the core requirement.
LangChain agents are built on LangGraph internally, but AgentBridge keeps these adapters separate so
teams can choose between direct app compatibility and explicit graph control.

For an existing compiled LangGraph, preserve it directly instead of rebuilding it:

```python
agent = LangGraphExtension.with_config(
    agent,
    native_graph=compiled_graph,
    stream_options={"stream_mode": "updates", "version": "v2"},
)
```

`native_graph` keeps the framework's own topology, nodes, checkpointers, stores, and middleware
intact. For generated graphs, pass `checkpointer=`, `store=`, and `cache=` directly through
`LangGraphExtension.config()`. `stream_options` controls native graph streaming, while
`native_options` remains the escape hatch for compile-time and newer runtime options.

Checkpointed graphs also expose native state inspection and time travel through the adapter:
`get_state()`, `get_state_history()`, `update_state()`, and `replay(checkpoint_id=...)`. These
helpers preserve the backend checkpoint payloads while AgentBridge supplies the session and
observability configuration.

When `LangChainExtension.config(callbacks=..., metadata=...)` is used, the adapter passes those
values into LangChain's native runtime config and includes a serializable `runtime_config` summary in
`RunResult.metadata`.

Enable LangSmith and Langfuse through the same AgentBridge configuration:

```python
agent = LangChainExtension.with_config(
    agent,
    observability={
        "tags": ["production", "refunds"],
        "run_name": "refund-agent",
        "langsmith": {
            "enabled": True,
            "project_name": "agentbridge-refunds",
        },
        "langfuse": {
            "enabled": True,
            "user_id": "customer-123",
        },
    },
)
```

LangSmith and Langfuse credentials remain environment/provider concerns. Install the optional
integration dependencies with `pip install 'agentbridge-langchain[observability]'`. AgentBridge
forwards native callbacks and metadata, adds Langfuse's `CallbackHandler` when enabled, and wraps
the invocation/stream in LangSmith's native tracing context. No API keys are stored in `AgentSpec`.
Langfuse callback options such as `release`, `version`, `environment`, `session_id`, `user_id`, and
`trace_id` are forwarded when supported by the installed SDK, with a fallback for older SDK lines.
Future or provider-specific callback fields can be passed through
`observability.langfuse.callback_options` without changing core.

LangSmith smoke coverage is available but disabled by default to avoid accidental network calls:

```bash
export AGENTBRIDGE_LANGSMITH_SMOKE=1
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=agentbridge-smoke
pytest plugins/agentbridge-langchain/tests/test_langsmith_smoke.py
```

The smoke test runs the offline LangChain adapter inside a LangSmith tracing context and verifies
that AgentBridge runtime metadata is preserved.

Run the combined credentialed observability lane with:

```bash
export AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1
export LANGSMITH_API_KEY=...
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
python examples/observability_smoke.py
```

The LangSmith check authenticates against `GET /info` without creating a run. The Langfuse check
executes the offline LangChain agent with the native Langfuse callback enabled, proving callback
construction and AgentBridge runtime wiring without depending on model-provider credentials.

Publish the same framework-neutral evaluation examples to LangSmith and run a hosted experiment:

```python
from agentbridge import EvaluationExample
from agentbridge_langchain.langsmith_evaluation import evaluate_on_langsmith

results = evaluate_on_langsmith(
    agent,
    backend="langchain",
    dataset=[EvaluationExample(input="Check order A123")],
    dataset_name="refund-regression",
    experiment_prefix="agentbridge-refund-v1",
)
```

The integration creates the dataset with `Client.create_dataset`, uploads examples with
`Client.create_examples`, and delegates execution/evaluator orchestration to the native
`Client.evaluate` API. Credentials, evaluator definitions, and hosted retention remain
LangSmith concerns; AgentBridge supplies the portable target and input mapping.

Prompt versioning is available without adding LangSmith to core:

```python
from agentbridge_langchain.langsmith_prompts import pull_prompt, push_prompt

prompt = pull_prompt("team/refunds:production")
commit = push_prompt(
    "team/refunds",
    prompt=prompt,
    tags=["production"],
    commit_description="Reviewed refund policy",
)
```

The native prompt object is returned unchanged so LangChain remains responsible for template
variables, message formatting, model-specific prompt behavior, and prompt serialization.

For direct Langfuse API operations without adding another framework abstraction:

```python
from agentbridge_langchain.langfuse_api import LangfuseAPIClient

client = LangfuseAPIClient()
health = client.request_json("GET", "/api/public/health")
```

The transport preserves arbitrary endpoint payloads and keeps Langfuse credentials in the
environment. Use `ingest_otlp()` for current OTLP/HTTP trace export, and use the native Langfuse
SDK callback for framework traces. `request_json()` remains available for API operations that are
not yet normalized.

Fetch and compile managed prompts through the same client:

```python
from agentbridge_langchain.langfuse_prompts import fetch_prompt

prompt = fetch_prompt(client, "refund-policy", label="production")
compiled = prompt.compile(customer_tier="gold")
langchain_template = prompt.get_langchain_prompt()
```

LangSmith deployment and control-plane API pages are exposed through an optional generic transport:

```python
from agentbridge_langchain.langsmith_api import LangSmithAPIClient

client = LangSmithAPIClient()
thread = client.request_json("POST", "/v1/threads", body={})
events = client.stream_events("POST", "/threads/THREAD_ID/runs/stream", body={})
```

The client supports arbitrary JSON endpoints and newline-delimited SSE responses, including
query parameters and raw payloads. `call()`/`acall()` provide one entry point for any current or
future LangSmith endpoint. It is intentionally native-only: endpoint-specific schemas,
hosted lifecycle behavior, and credentials remain LangSmith concerns rather than being faked as
portable AgentBridge semantics.

For deployment automation, use the typed control-plane client. It keeps the workspace identity
explicit and covers deployment creation, updates, deletion, revision listing/creation/status,
and free-deployment quota checks:

```python
from agentbridge_langchain import LangSmithControlPlaneClient

control_plane = LangSmithControlPlaneClient(tenant_id="workspace-id")
deployment = control_plane.create_deployment(body={"name": "refunds"})
revisions = control_plane.list_revisions(deployment["id"])
```

Set `LANGSMITH_API_KEY`, `LANGSMITH_TENANT_ID`, and optionally
`LANGSMITH_CONTROL_PLANE_URL` for hosted or self-hosted control-plane environments.

For a deployed LangGraph or Agent Server graph, use the normalized remote client:

```python
from agentbridge_langchain import RemoteGraphClient

remote = RemoteGraphClient(client)
result = remote.run(
    thread_id="thread-123",
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Check order A123"}]},
)
```

The native thread/run API remains available through `LangSmithAPIClient` for endpoints that need
provider-specific payloads.

SSE responses retain their native `event` and `id` fields, including multiline JSON data.
`RemoteGraphClient` places these in `AgentEvent.metadata.native_event` and `sse_id`; namespaced
message streams preserve their original namespace while emitting normalized message events.
Data-only JSON event envelopes retain their existing response shape. The default HTTP transport
streams complete frames as they arrive; async calls read one frame at a time off the event loop.
Injected legacy transports returning bytes remain buffered. Use `closing()`/`aclosing()` to
release the response when ending a stream early. See the
[remote streaming guide](../../docs/remote_streaming.md) for examples and cancellation semantics.
Remote execution accepts `run_options` for native checkpoint, context, durability, interrupt,
stream-mode, resumability, and governance fields. `reconnect()`/`areconnect()` join an existing run
stream with `Last-Event-ID` and selected stream modes, without starting a replacement run.
Remote results retain the last full `values` state when an empty `end` frame arrives. Error events
set `RunResult.metadata.status` to `error` and do not synthesize a success completion event.
Dynamic approval interrupts and empty static-breakpoint updates report `status="interrupted"`
with preserved payloads/IDs. Interrupted and failed runs retain terminal stream frames as workflow
events. `examples/remote_graph_approval.py` verifies checkpointed approval/resume through a real
local LangGraph and an injected SSE transport.

The same client exposes assistant lifecycle and graph/schema/subgraph/version introspection, thread
search/history/patch/copy/interrupts, run listing/events/join/cancellation, run feedback, long-term
store operations, thread-state helpers, Fleet/Managed Deep Agent CRUD, managed-agent thread metadata,
and registered MCP-server lifecycle operations for common LangGraph and LangSmith workflows while
retaining `request_json()`/`call()` for the complete native API surface. Cursor-safe `iter_agents()`
and `iter_fleet_threads()` helpers follow native `next_cursor` pagination, and
`list_trigger_templates()` exposes Fleet trigger schemas. Use `path_prefix="/v1/deepagents"` for
Managed Deep Agents; Fleet agents use the default `/v1/fleet` prefix.

Stateless/background execution is available through `create_background_run()`, blocking execution
through `create_run_wait()`, native SSE through `create_run_stream()`, and batch submission through
`create_run_batch()`. Scheduled execution is covered by `create_cron()`,
`create_thread_cron()`, `search_crons()`, `count_crons()`, `get_cron()`, `update_cron()`, and
`delete_cron()`; native request bodies are preserved so fields such as context, durability,
stream modes, webhooks, interrupts, and completion policy are not discarded. Agent-to-Agent
JSON-RPC and stateless MCP transport are available through `a2a_json_rpc()`, `a2a_stream()`,
`mcp_get()`, `mcp_post()`, and `mcp_terminate()`.
Agent Server health, version metadata, local API docs, system metrics, and persistent thread
subscriptions are available through `health_check()`, `server_info()`, `api_documentation()`,
`system_metrics()`, and `join_thread_stream()`. Checkpoint reads and retention controls are
available through `get_thread_state_at_checkpoint()`, `get_thread_state_at_checkpoint_body()`,
and `prune_threads()`.

Remote state and approval flows are available through `RemoteGraphClient.state()`,
`RemoteGraphClient.update_state()`, and `RemoteGraphClient.resume()`, with matching async
`acreate_thread()`, `astate()`, `aupdate_state()`, and `aresume()` methods. Resume sends the
Agent Server's top-level `command` field with `input=None`, so the decision reaches the paused
interrupt rather than becoming graph state:

```python
run = remote.resume(
    thread_id="thread-123",
    assistant_id="agent",
    resume_value={"decisions": [{"type": "approve"}]},
)
# resume_value is defined by the graph's interrupt; parallel interrupts can use an ID/value map.
```

For streaming approval flows, use `client.create_thread_run(..., input=None,
command={"resume": decision}, stream=True)` or its async counterpart. The async client also
exposes `adelete_run()` for persisted run cleanup.

Langfuse evaluation publishing
is available through `publish_langfuse_dataset()` and `publish_report_scores()`.

For an option that AgentBridge has not normalized yet, pass it explicitly:

```python
agent = LangChainExtension.with_config(
    agent,
    native_options={"new_langchain_option": value},
)
```

The option is included in adapter diagnostics and is forwarded unchanged to native
`create_agent`. This is the compatibility path for upstream additions while a normalized
AgentBridge contract is being designed.

When `stream_agent(..., backend="langchain")` is used, the adapter first tries LangChain's event
streaming API and normalizes message deltas, tool-call chunks, completed tool calls, tool results,
and custom updates into `AgentEvent` values. If a LangChain runtime only supports classic
`stream()`, the adapter requests `messages`, `updates`, and `custom` stream modes before falling back
to the runtime's default stream signature.

When `LangChainExtension.config(memory=..., retrievers=...)` is used, AgentBridge records those hints
in `RunResult.metadata["extension_summary"]`. Portable memory/retriever semantics are not claimed
unless the caller supplies native LangChain objects such as `checkpointer` and `store`, which the
adapter forwards directly into `create_agent`.

See [examples/langchain_native_memory_retriever.py](../../examples/langchain_native_memory_retriever.py)
for an offline example that passes a real LangChain `BaseRetriever`, LangGraph `InMemorySaver`, and
LangGraph `InMemoryStore` through `LangChainExtension`.

## Local Development Without Installing

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="agentbridge_langchain.adapter"
agentbridge plugins
```

## Implementation Checklist

- Add the framework package dependency to `pyproject.toml`.
- Update `Adapter.capabilities()` with honest support metadata.
- Implement `compile()` by translating `AgentSpec` to native framework objects.
- Implement `run()` by returning a normalized `RunResult`.
- Implement `stream()` if the backend supports streaming.
- Add contract tests for every capability marked `full`.
- Run `agentbridge conformance --backend langchain` before publishing.
- Document adopted and verified framework versions.
