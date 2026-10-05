# LangChain Ecosystem Coverage

AgentBridge tracks the LangChain ecosystem from the official recursive
documentation index at [`docs.langchain.com/llms.txt`](https://docs.langchain.com/llms.txt).
The current URL snapshot is [`docs/upstream/langchain-pages.json`](upstream/langchain-pages.json).
The page-level decision ledger is [`docs/upstream/langchain-coverage.json`](upstream/langchain-coverage.json).
It is an audit trail: every page has an explicit status, owner, and next action.
Executable native conformance is tracked in [`docs/upstream/langchain-conformance.json`](upstream/langchain-conformance.json).

## Coverage Contract

Every inventoried capability must have one explicit outcome:

| Status | Meaning |
| --- | --- |
| `core` | Expressed through framework-neutral AgentBridge types and behavior. |
| `extension` | Exposed through a LangChain or LangGraph AgentBridge extension, with adapter tests. |
| `native_only` | Available through the preserved raw backend object or native options, but not normalized yet. |
| `planned` | A documented gap with an implementation slice and acceptance tests still to add. |
| `unsupported` | Deliberately outside the current product boundary, with a reason recorded. |

An adapter must not silently discard an upstream option. If a feature cannot be
normalized, the adapter should preserve it through an extension configuration,
raw backend escape hatch, or an explicit diagnostic.

## Current Surface

| LangChain area | AgentBridge surface | Current status |
| --- | --- | --- |
| LangChain `create_agent`, tools, retriever tools, structured output, middleware, streaming, runtime context | `AgentSpec`, `ToolSpec`, explicit `LangChainExtension.config(retriever_tools=...)`, `LangChainExtension.context_schema`, native `context=` invocation, plugin configuration, and normalized events | `extension` |
| LangChain human-in-the-loop middleware and checkpointed pauses | Native `HumanInTheLoopMiddleware`, checkpointer pass-through, normalized interrupted workflow event | `extension` |
| LangChain MCP tools | Native MCP tool objects supplied through `LangChainExtension(mcp_tools=...)`; optional MCP adapter packages remain external | `extension` |
| Newly added LangChain `create_agent` options | Guarded `LangChainExtension.native_options` pass-through with diagnostics and offline conformance | `native_only` until normalized |
| LangGraph state graphs, validated conditional routing, checkpoints, interrupts, resume, state history/time travel, custom checkpointers, stores, caches, retrievers, native model invocation, sync/async tools, per-run runtime config, batch execution, and native graph escape hatches | Built-in LangGraph adapter, `LangGraphExtension.conditional_routing()`, `get_state`/`get_state_history`/`update_state`/`replay`, native `batch()`/`abatch()` dispatch, async `ToolSpec` execution in generated graphs, `runtime_config` forwarding for native `RunnableConfig` controls, explicit persistence fields, `LangGraphConfig.model`, `LangGraphConfig.native_graph`, and guarded `native_options` compile pass-through | `extension` |
| LangSmith tracing, metadata, tags, run names, sessions, trace context | Shared observability helpers and LangChain/LangGraph config; runtime-config conformance is executable offline | `extension` |
| Cross-provider trace correlation | Core `observability_metadata()` maps session, user, project, and trace identifiers consistently for LangChain and LangGraph | `extension` |
| OpenAI-compatible, NVIDIA NIM, OpenRouter, local Ollama, Anthropic, Google GenAI/Vertex AI, Mistral, Groq, Cohere, Bedrock, Fireworks, Hugging Face, xAI, Azure OpenAI, and arbitrary LangChain model routes | Lazy provider factories forward model options, or accept any native model object, without moving credentials or provider packages into core | `extension` |
| LangChain model retries and fallbacks | `LangChainExtension.config(model_retry=..., model_fallbacks=[...])` applies native runnable resilience wrappers to supplied model objects | `extension` |
| Langfuse LangChain callback integration | Lazy callback integration in the optional LangChain plugin | `extension` |
| LangSmith assistants, graph/schema/subgraph/version introspection, threads, state/checkpoints/pruning, runs, stateless/background/batch execution, crons, interrupts, long-term store, Fleet/Managed Agents, Agent Auth connections, platform tools, A2A JSON-RPC, MCP transport, system health/info/docs/metrics, and SSE APIs | Typed sync/async lifecycle/search/patch/copy/events/join/cancellation/checkpoint/pruning/stateless-run/cron/store/agent/connection/tool-registry/protocol/system helpers over dependency-free JSON/SSE transports plus `RemoteGraphClient` resume/state helpers | `extension` |
| Langfuse traces, OTLP ingestion, observations, scores, metrics, prompts, experiments, datasets, dataset items, and evaluation reports | Typed sync/async query and lifecycle helpers, cursor-safe dataset/dataset-item/prompt/experiment/observation iteration, async dataset deletion and item update parity, `LangfusePrompt`, plus `langfuse_evaluation` bridge over the dependency-free JSON/SSE/raw transport | `extension` |
| Remote LangGraph/Agent Server runs | `RemoteGraphClient` maps native thread/run SSE events to `AgentEvent` and `RunResult` | `extension` |
| AgentCore Memory/Gateway bindings | `LangChainExtension.agentcore` plus credential-free conformance and AgentCore plugin contracts | `extension` |
| Async LangChain execution | Native `ainvoke` path normalized through `arun_agent` | `extension` |
| LangChain batch and async-batch execution | Public `batch_agent()`/`abatch_agent()` with native `batch()`/`abatch()` dispatch and ordered fallback | `extension` |
| Async normalized streaming | Public `astream_agent()` contract, LangChain native `astream`/`astream_events`, and LangGraph native `astream` with fallback | `extension` |
| LangSmith datasets, examples, evaluators, prompts, experiments, monitoring, feedback, REST API, governance, and deployment control plane | Evaluation contract, typed dataset/example CRUD and cursor iteration, bulk example deletion, OpenAI fine-tuning export, dataset version reads/diffs, split updates, public share/unshare, regression deltas, shared-dataset examples with runs, annotation queues and human-review run assignment, feedback configuration CRUD and presigned feedback tokens, SDK or dependency-free API-client dataset publishing/evaluation bridge, prompt pull/push helpers, typed feedback/thread/MCP/deployment clients, and generic API transport; remaining hosted APIs stay native | `extension` |
| Deep Agents, sandboxes, filesystem backends, permissions, skills, interpreters | External `agentbridge-deepagents` plugin maps `create_deep_agent`, native backends, skills, memory, subagents, permissions, HITL, persistence, structured output, and native options | `extension` |
| LangChain deployment, Agent Server, Studio, Fleet, Managed Deep Agents, and listeners | Typed LangSmith deployment/revision control-plane operations including redeploy, interruption, bulk deletion, logs, and tier updates; v2 listener lifecycle; remote Agent Server thread/run client, Fleet/Managed Deep Agent CRUD, and generic transport; Studio UI and hosted publishing remain native | `extension` or `native_only` depending on page |
| LangChain LLM Gateway and provider administration | Model routing remains delegated to LiteLLM-style strings; no LangSmith gateway control plane | `native_only` |
| LangChain TypeScript documentation | Python SDK scope | `unsupported` for the current SDK |

## Refresh And Review

Refresh the snapshot intentionally after reviewing new pages:

```bash
python scripts/crawl_langchain_docs.py \
  --output docs/upstream/langchain-pages.json
python scripts/build_langchain_coverage.py
```

Check for upstream drift without changing the repository:

```bash
python scripts/crawl_langchain_docs.py \
  --check --fail-on-drift
python scripts/build_langchain_coverage.py --check
```

When drift is found, review each added or removed page, update the capability
ledger and adapter tests, then refresh the snapshot in the same change. The
weekly `Upstream Compatibility` workflow performs this check and uploads the
inventory when it fails.

## Next LangChain Slices

Run the credential-free native conformance lane:

```bash
uv run pytest -q plugins/agentbridge-langchain/tests/test_native_conformance.py
```

This lane proves translation and normalization for the current native-only surface. It does not
claim provider, hosted deployment, or remote LangSmith behavior; those require separate credentialed
smoke lanes. It now also checks native async execution/streaming contracts, AgentCore binding
preservation, and LangSmith/Langfuse provider option preservation without loading credentials.

Run the explicitly credentialed observability lane:

```bash
AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 \
  LANGSMITH_API_KEY=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
  python examples/observability_smoke.py
```

The lane reports LangSmith API authentication and Langfuse callback/runtime wiring separately from
the credential-free native conformance report.

The first pass is intentionally ordered by value to application developers:

1. Complete LangChain/LangGraph model, tool, middleware, streaming, structured-output, memory, retrieval, and human-in-the-loop option forwarding.
2. Use the new core `EvaluationExample`/`EvaluationReport` contract for provider-neutral offline evaluation, then add LangSmith dataset/evaluator publishing.
3. Expand LangSmith prompt/version and experiment integration while preserving raw clients; Langfuse prompt retrieval/version compilation is now executable.
4. Expand OpenTelemetry-compatible trace export and first-class Langfuse/LangSmith trace correlation.
5. Add typed remote LangGraph/Agent Server operations beyond thread/run streaming, including
   assistants, checkpoints, store access, interrupts, and deployment lifecycle APIs.
6. Expand the Deep Agents plugin with credentialed sandbox/interpreter lanes and hosted deployment checks; the local translation contract is now executable.

This order keeps the core dependency-free while still giving every upstream
feature a visible place in the roadmap and compatibility review.
