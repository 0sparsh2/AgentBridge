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
- Native `HumanInTheLoopMiddleware` pauses are surfaced as normalized workflow interrupt events and
  preserve the native checkpoint state for the caller's approval/resume flow.
- Native MCP adapter tools can be supplied with `LangChainExtension(mcp_tools=[...])` and are
  forwarded unchanged to `create_agent`; `langchain-mcp-adapters` remains optional.
- OpenAI-compatible native models can use `model_provider="openai"` and `model_options={...}`
  for NVIDIA NIM, OpenRouter, or internal gateways; install the optional `openai` extra.
- A guarded `native_options` escape hatch for newly released LangChain `create_agent` options;
  AgentSpec-owned identity/model/tool fields cannot be overridden.
- LangSmith tracing context and Langfuse callback integration through one AgentBridge observability config.
- LangSmith dataset publishing and hosted evaluation through an optional integration module.
- Langfuse callback wiring plus a dependency-free JSON/SSE API transport for ingestion and export endpoints.
- Streaming normalization for LangChain `stream_events(..., version="v3")` event envelopes and
  `stream(..., stream_mode=["messages", "updates", "custom"], version="v2")` chunks.

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
environment. Use the native Langfuse SDK callback for framework traces and this client for API
operations that are not yet normalized.

LangSmith deployment and control-plane API pages are exposed through an optional generic transport:

```python
from agentbridge_langchain.langsmith_api import LangSmithAPIClient

client = LangSmithAPIClient()
thread = client.request_json("POST", "/v1/threads", body={})
events = client.stream_events("POST", "/threads/THREAD_ID/runs/stream", body={})
```

The client supports arbitrary JSON endpoints and newline-delimited SSE responses, including
query parameters and raw payloads. It is intentionally native-only: endpoint-specific schemas,
hosted lifecycle behavior, and credentials remain LangSmith concerns rather than being faked as
portable AgentBridge semantics.

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
