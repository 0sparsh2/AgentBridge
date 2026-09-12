# agentbridge-langchain

AgentBridge adapter plugin for `langchain`.

## Adopted Framework Version

- Native package: `langchain`
- Adopted range: `>=1.4,<2`
- Verified locally: `1.4.0`
- Status: partial native adapter

## Target Capabilities

- Direct LangChain `create_agent` compatibility alongside the built-in LangGraph backend.
- `ToolSpec` to `StructuredTool` mapping.
- Middleware, callbacks, memory, and retriever configuration through `LangChainExtension`.
- LangSmith/callback-style tracing metadata.

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
