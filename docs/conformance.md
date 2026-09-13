# Conformance

AgentBridge conformance checks verify the minimum contract every adapter must satisfy before it can
claim baseline compatibility.

Run all discovered backends:

```bash
agentbridge conformance
```

Run one backend:

```bash
agentbridge conformance --backend langchain
agentbridge conformance --backend google_adk --json
```

## What It Checks

- Capability metadata exists and names the selected backend.
- A simple no-tool agent compiles and returns normalized output.
- Streaming returns normalized events and ends with `complete`.
- Sync tools work when the adapter advertises `tools.sync: full`.
- Structured output works when the adapter advertises `structured_output: full`.
- Run diagnostics metadata exists when the adapter advertises `observability.diagnostics: full`.

Passing conformance means the adapter satisfies the portable AgentBridge contract. It does not mean
every native framework feature is fully abstracted.

## Offline Plugin Models

Cloud-backed frameworks often require provider packages, API keys, or cloud credentials. Their
AgentBridge plugins therefore provide a plugin-only `agentbridge/offline` model string for
conformance. This string is for tests and contract checks only.

Current offline conformance paths:

| Backend | Native path still exercised | Offline model implementation |
| --- | --- | --- |
| `openai_agents` | OpenAI Agents SDK `Agent` and `Runner` | Local SDK `Model` subclass |
| `strands` | Strands `Agent` event loop | Local Strands `Model` subclass |
| `langchain` | LangChain `create_agent` graph | Local LangChain `BaseChatModel` subclass |
| `google_adk` | Google ADK `Agent` and `Runner` | Local ADK `BaseLlm` subclass |

## Current Snapshot

As of the latest local sweep, `agentbridge conformance` passes for every discovered backend.

| Backend | Baseline | Tools | Streaming | Structured Output | Diagnostics |
| --- | --- | --- | --- | --- | --- |
| `google_adk` | pass | pass | pass | pass | pass |
| `langchain` | pass | pass | pass | pass | pass |
| `langgraph` | pass | pass | pass | skipped, not advertised as `full` | pass |
| `mock` | pass | pass | pass | skipped, not advertised as `full` | skipped, not advertised as `full` |
| `openai_agents` | pass | pass | pass | pass | pass |
| `pydantic_ai` | pass | pass | pass | pass | skipped, not advertised as `full` |
| `strands` | pass | pass | pass | pass | pass |

Production users should keep using normal framework model identifiers, such as `openai/gpt-5`,
`anthropic/claude-sonnet`, or framework-native model objects. Adapter-specific offline models are not
provider-routing abstractions.

## Relationship To Feature Parity

Conformance is the floor. Feature parity is tracked through capability reports, adapter docs, and
framework-specific issues.

Use these commands together:

```bash
agentbridge conformance --backend langchain
agentbridge coverage-report --backend langchain --markdown
agentbridge extensions langchain --json
```

Only upgrade a capability to `full` when a native-framework test proves the behavior. Keep framework
nuance as `extension` or `native_only` when forcing it into the common `AgentSpec` would be
misleading.
