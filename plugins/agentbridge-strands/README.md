# agentbridge-strands

AgentBridge adapter plugin for `strands`.

## Adopted Framework Version

- Native package: `strands-agents`
- Adopted range: `>=1.55,<2`
- Verified locally: `1.55.1`
- Status: partial native adapter

## Target Capabilities

- Agent and tool mapping.
- MCP client/tool-provider pass-through through `StrandsExtension`; plain string labels are kept as
  metadata hints.
- Native hooks, plugins, interventions, session managers, memory managers, context managers, retry
  strategies, checkpointing, sandbox, storage, and background-task options through
  `StrandsExtension`.
- Structured output model pass-through with native contract tests.
- Best-effort hook, intervention, and guardrail lifecycle event normalization.
- Run diagnostics for events, guardrails, hooks, interventions, MCP clients, usage, interrupts, and
  checkpoints.
- Trace attributes and AWS/serverless deployment metadata summaries.

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

`StrandsExtension` forwards native Strands `Agent` constructor options when supplied and reports a
serializable `extension_summary` in `RunResult.metadata`. Native MCP client/tool-provider objects in
`mcp_clients` are appended to the Strands `tools` list; hook, intervention, and guardrail-shaped
stream events are normalized as workflow events with lifecycle phases. Plain string labels,
guardrail labels, and deployment targets are recorded as extension metadata until AgentBridge has
deeper native no-network tests for those execution paths.

Deployment metadata can be recorded without installing AWS deployment tooling:

```python
from agentbridge import AgentSpec
from agentbridge.extensions.strands import StrandsExtension

agent = StrandsExtension.with_config(
    AgentSpec(
        name="support_agent",
        instructions="Help customers with refunds.",
        model="bedrock/us.anthropic.claude-sonnet-4-5",
    ),
    deployment_target="agentcore",
    deployment={
        "runtime": "bedrock-agentcore",
        "entrypoint": "app:agent",
        "region": "us-east-1",
        "session_header": "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id",
        "trace_headers": ["X-Amzn-Trace-Id", "traceparent", "baggage"],
        "observability": {"cloudwatch": True, "xray": True},
    },
)
```

This metadata is preserved in `RunResult.metadata["extension_summary"]["deployment"]` for review,
comparison, and future deploy tooling. It does not call AgentCore, build containers, provision IAM,
or publish runtime endpoints.

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
