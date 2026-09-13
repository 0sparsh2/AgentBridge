# agentbridge-openai-agents

AgentBridge adapter plugin for `openai_agents`.

## Adopted Framework Version

- Native package: `openai-agents`
- Adopted executable range: `>=0.20,<0.21`
- Latest observed during scaffolding: `0.22.2` on 2026-09-11
- Status: partial native adapter; `0.22.x` verification is blocked until AgentBridge's LiteLLM/OpenAI dependency path supports `openai>=3`

## Target Capabilities

- Agent and tool mapping.
- Native handoffs, handoff descriptions, MCP servers/config, prompts, model settings, hooks,
  tool-use behavior, reset behavior, and input/output guardrails through `OpenAIAgentsExtension`.
- Structured output through native SDK `output_type` and typed `final_output`.
- Runner context, max turns, run hooks, run config, error handlers, previous response IDs,
  conversation/session options, and session objects through `OpenAIAgentsExtension`.
- Approval policy, approval interruption, guardrail diagnostic, handoff item, and tracing metadata
  summaries.
- Application-owned approval request stores and the backend-neutral `ApprovalQueue` helper through
  `OpenAIAgentsExtension.approval_store`.
- Raw run/result preservation.

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

The conformance runner uses the plugin-only `agentbridge/offline` model string. That path still
compiles an OpenAI Agents SDK `Agent` and runs through the native `Runner`, but uses a tiny local
SDK `Model` implementation so contract checks do not require paid API credentials.

`OpenAIAgentsExtension` forwards native SDK options when supplied and reports serializable
`runner_kwargs`, `extension_config`, `extension_summary`, and `run_diagnostics` metadata on
`RunResult`.

`run_diagnostics` preserves safe summaries of SDK result surfaces that are important for production
UIs and audits:

- `interruptions`: pending approval interruptions reported by the SDK.
- `resumable`: whether interruptions exist and the result exposes `to_state()`.
- `state_type`: the SDK state object's type name, when it can be captured safely.
- `last_agent`, `last_response_id`, and `raw_responses_count`: continuation and debugging hints.
- `guardrails`: input, output, tool-input, and tool-output guardrail result summaries.
- `approval_store`: whether approval requests were written to an application-owned store.

Approval, guardrail, and handoff run items are also normalized as `AgentEvent(type="workflow")`.
If `OpenAIAgentsExtension.approval_store` is supplied, approval interruptions are written as safe
records containing the backend, pending status, interruption summary, SDK state snapshot summary,
last response ID, and last agent summary. Store objects can expose
`record_approval_request(record)`, `save(record)`, or `append(record)`.

For a lightweight backend-neutral queue, pass `agentbridge.ApprovalQueue`:

```python
from agentbridge import AgentSpec, ApprovalQueue
from agentbridge.extensions.openai_agents import OpenAIAgentsExtension

approval_queue = ApprovalQueue()
agent = OpenAIAgentsExtension.with_config(
    AgentSpec(
        name="support_agent",
        instructions="Use approvals for refund tools.",
        model="openai/gpt-5",
    ),
    approval_policy={"issue_refund": "required"},
    approval_store=approval_queue,
)

# After a run records an interruption:
pending = approval_queue.pending(backend="openai_agents")
resume_payload = approval_queue.approve(
    pending[0]["id"],
    response={"approved": True},
    reviewer="support-lead",
)
```

The queue returns a portable resume payload containing the approval decision, SDK state summary,
last response ID, and last agent summary. AgentBridge still does not execute the native OpenAI
Agents SDK resume call for you; use the payload, preserved raw result, or SDK state with
application-owned approval workflows.

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
