# LangChain Ecosystem Coverage

AgentBridge tracks the LangChain ecosystem from the official recursive
documentation index at [`docs.langchain.com/llms.txt`](https://docs.langchain.com/llms.txt).
The current URL snapshot is [`docs/upstream/langchain-pages.json`](upstream/langchain-pages.json).
The page-level decision ledger is [`docs/upstream/langchain-coverage.json`](upstream/langchain-coverage.json).
It is an audit trail: every page has an explicit status, owner, and next action.

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
| LangChain `create_agent`, tools, structured output, middleware, streaming, runtime context | `AgentSpec`, `ToolSpec`, LangChain plugin configuration and normalized events | `extension` |
| LangGraph state graphs, routing, checkpoints, interrupts, resume, stores, retrievers | Built-in LangGraph adapter and `LangGraphConfig` | `extension` |
| LangSmith tracing, metadata, tags, run names, sessions, trace context | Shared observability helpers and LangChain/LangGraph config | `extension` |
| Langfuse LangChain callback integration | Lazy callback integration in the optional LangChain plugin | `extension` |
| LangSmith datasets, evaluators, prompts, experiments, monitoring, REST API, governance | No normalized SDK surface yet; tracked by the page inventory | `planned` |
| Deep Agents, sandboxes, filesystem backends, permissions, skills, interpreters | No dedicated adapter; raw model/tool primitives remain usable | `planned` |
| LangChain deployment, Agent Server, Studio, Fleet, Managed Deep Agents | No hosted control plane in AgentBridge v0/v1 | `planned` or `unsupported` depending on page |
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

The first pass is intentionally ordered by value to application developers:

1. Complete LangChain/LangGraph model, tool, middleware, streaming, structured-output, memory, retrieval, and human-in-the-loop option forwarding.
2. Use the new core `EvaluationExample`/`EvaluationReport` contract for provider-neutral offline evaluation, then add LangSmith dataset/evaluator publishing.
3. Add prompt/version and experiment integration while preserving raw LangSmith clients.
4. Add OpenTelemetry-compatible trace export and first-class Langfuse/LangSmith trace correlation.
5. Add deployment and Deep Agents plugins only after their hosted/runtime boundaries are defined.

This order keeps the core dependency-free while still giving every upstream
feature a visible place in the roadmap and compatibility review.
