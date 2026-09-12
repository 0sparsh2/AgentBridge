# agentbridge-openai-agents

AgentBridge adapter plugin for `openai_agents`.

## Adopted Framework Version

- Native package: `openai-agents`
- Adopted executable range: `>=0.20,<0.21`
- Latest observed during scaffolding: `0.22.2` on 2026-09-11
- Status: partial native adapter; `0.22.x` verification is blocked until AgentBridge's LiteLLM/OpenAI dependency path supports `openai>=3`

## Target Capabilities

- Agent and tool mapping.
- Handoffs through `OpenAIAgentsExtension`.
- Guardrails through `OpenAIAgentsExtension`.
- Human approval flows.
- Native tracing and raw run/result preservation.

## Dependency Note

`openai-agents` 0.22.x requires `openai>=3`, while the current AgentBridge core dependency path through LiteLLM uses `openai<3`. This plugin therefore adopts `openai-agents>=0.20,<0.21` first, because that line remains compatible with the current core environment. Upgrade the range only when the dependency conflict is resolved and contract tests pass.

## Install

```bash
pip install -e .
```

## Verify Discovery

```bash
agentbridge plugins
agentbridge list-backends
agentbridge inspect-backend openai_agents --json
agentbridge conformance --backend openai_agents
```

## Local Development Without Installing

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="agentbridge_openai_agents.adapter"
agentbridge plugins
```

## Implementation Checklist

- Add the framework package dependency to `pyproject.toml`.
- Update `Adapter.capabilities()` with honest support metadata.
- Implement `compile()` by translating `AgentSpec` to native framework objects.
- Implement `run()` by returning a normalized `RunResult`.
- Implement `stream()` if the backend supports streaming.
- Add contract tests for every capability marked `full`.
- Run `agentbridge conformance --backend openai_agents` before publishing.
- Document adopted and verified framework versions.
