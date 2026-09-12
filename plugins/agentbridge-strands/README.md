# agentbridge-strands

AgentBridge adapter plugin for `strands`.

## Adopted Framework Version

- Native package: `strands-agents`
- Adopted range: `>=1.55,<2`
- Verified locally: `1.55.1`
- Status: partial native adapter

## Target Capabilities

- Agent and tool mapping.
- MCP clients through `StrandsExtension`.
- Hooks for lifecycle, guardrail, approval, and streaming events.
- Structured output model pass-through.
- Trace attributes and AWS/serverless deployment metadata.

## Install

```bash
pip install -e .
```

## Verify Discovery

```bash
agentbridge plugins
agentbridge list-backends
agentbridge inspect-backend strands --json
agentbridge conformance --backend strands
```

The conformance runner uses the plugin-only `agentbridge/offline` model string. That path still
compiles a Strands `Agent` and runs through the native event loop, but uses a tiny local Strands
`Model` implementation so contract checks do not require cloud credentials.

## Local Development Without Installing

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="agentbridge_strands.adapter"
agentbridge plugins
```

## Implementation Checklist

- Add the framework package dependency to `pyproject.toml`.
- Update `Adapter.capabilities()` with honest support metadata.
- Implement `compile()` by translating `AgentSpec` to native framework objects.
- Implement `run()` by returning a normalized `RunResult`.
- Implement `stream()` if the backend supports streaming.
- Add contract tests for every capability marked `full`.
- Run `agentbridge conformance --backend strands` before publishing.
- Document adopted and verified framework versions.
