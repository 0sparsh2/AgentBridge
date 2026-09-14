# Scenario Reports

Scenario reports are the long-term proof format for AgentBridge. They should show more than a
simple hello-world prompt: each report should demonstrate framework-specific nuance, migration
between frameworks, model/provider routing choices, and normalized outputs.

## Why They Matter

AgentBridge should eventually answer practical questions such as:

- What happens when the same refund agent moves from LangChain to LangGraph?
- Which framework-specific features were preserved, downgraded, or left native-only?
- Do outputs, events, tool calls, and diagnostics keep the same app-facing shape?
- Which model routing paths work for hosted APIs, local models, and OpenAI-compatible providers?

## Required Report Shape

Every deep scenario report should include:

- Scenario name and business goal.
- Source framework and target framework.
- `AgentSpec` fields used by both frameworks.
- Backend extension fields used for deep framework nuance.
- Model routing options tested or documented.
- Expected output shape and normalized `RunResult` fields.
- Expected streaming/event shape when relevant.
- Capability deltas, including `full`, `partial`, `extension`, `native_only`, and `unsupported`.
- Migration notes describing what was preserved, changed, or intentionally left native.
- Validation commands and whether they require credentials.

## Framework Nuance Examples

Reports should intentionally exercise non-trivial framework features:

- LangGraph: graph routing, checkpointing, interrupts, resume, conditional edges, state schemas.
- LangChain: LCEL runnable chains, middleware, callbacks, retrievers, stores, LangSmith tracing.
- Pydantic AI: typed outputs, Pydantic validation, validation retries, tool timeouts, dependency injection boundaries.
- OpenAI Agents SDK: handoffs, guardrails, approvals, Runner options, tracing and run items.
- Strands Agents: MCP clients, hooks, interventions, guardrail traces, AgentCore deployment metadata.
- Google ADK: session services, memory services, artifact services, sub-agents, eval/deployment metadata.
- CrewAI: roles, goals, tasks, crews, delegation, memory, human input when verified in a compatible environment.

## Model Routing Examples

AgentBridge does not build a second model-provider abstraction in v0. Model routing should stay in
LiteLLM-style model strings, backend-native model settings, or provider-compatible endpoint config.
Reports should still document the tested shapes:

| Provider Type | Example Shape | Notes |
| --- | --- | --- |
| OpenAI API | `openai/gpt-5` | Default docs/example shape. |
| Anthropic API | `anthropic/claude-sonnet` | Routed by backend/LiteLLM-compatible model handling. |
| Google API | `google/gemini` | Especially relevant for Google ADK. |
| Local model | `ollama/llama3.1` or backend-native local model object | Should be marked credential-free only when actually tested. |
| OpenRouter | `openrouter/openai/gpt-4o-mini` plus endpoint/API-key config | OpenAI-compatible endpoint behavior should be documented per backend. |
| NVIDIA NIM | `openai/deepseek-ai/deepseek-v4-flash-0731` plus NIM base URL/API-key config | Treat as OpenAI-compatible unless a backend has native NIM support. |
| Custom gateway | `openai/custom-model` plus backend-native base URL | Useful for internal gateways and local OpenAI-compatible servers. |

## Example Report Matrix

| Scenario | Source | Target | Deep Features | Model Routes | Default Test Mode |
| --- | --- | --- | --- | --- | --- |
| Refund approval migration | OpenAI Agents | LangGraph | approvals, queue payloads, handoffs, guardrails, MCP, interrupts, checkpoint resume | `agentbridge/offline`, `openai/gpt-5`, OpenRouter-compatible | Offline output comparison; native resume remains credentialed |
| Typed refund validation | Pydantic AI | LangGraph | typed outputs, validation retries, tool timeouts, graph checkpointing | `test`, hosted/local/OpenAI-compatible | Offline output comparison |
| Prototype crew migration | CrewAI | LangGraph | roles, goals, tasks, crews, delegation, memory, human input | hosted/local/OpenAI-compatible | Offline mock/LangGraph comparison; native CrewAI remains dependency-gated |
| Support RAG migration | LangChain | LangGraph | retriever, store, callbacks, graph topology | `agentbridge/offline`, local Ollama-style model | Offline output comparison |
| AWS production path | Strands | LangGraph | MCP, guardrails, hooks, interventions, session/memory managers, AgentCore metadata | Bedrock/NIM/OpenAI-compatible/local | Offline output comparison; live AgentCore remains credentialed |
| Google enterprise path | Google ADK | LangGraph | session, memory, artifacts, credentials, callbacks, sub-agents, eval labels, deployment metadata | `google/gemini`, local/offline/OpenAI-compatible | Offline output comparison; live eval/deploy remains credentialed |

## Acceptance Bar

A scenario report is not complete until it has:

- A runnable or inspectable example under `examples/` or `docs/examples/`.
- No-key default behavior for CI.
- Clear instructions for optional credentialed runs.
- Assertions or golden output where possible.
- A note explaining which framework nuances are fully tested versus metadata-only.

## Suite Runner

Run the aggregate report suite when changing report shape, model-route coverage, or framework
extension examples:

```bash
python examples/scenario_report_suite.py
```

The suite indexes every deep scenario report, records source/target framework coverage, and summarizes
which backend comparisons were available in the current environment.

## Credentialed Smoke Matrix

No scenario report should contact a paid API, local model server, or third-party gateway by default.
Credentialed smoke checks are double gated:

```bash
AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 python examples/credentialed_smoke_matrix.py
```

Then provide the route-specific credentials shown by the matrix, such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_BASE_URL`, `OPENROUTER_API_KEY`,
`NVIDIA_NIM_API_KEY` plus `NVIDIA_NIM_API_BASE` or `NVIDIA_NIM_BASE_URL`, optional `NVIDIA_MODEL`,
or the custom gateway environment variables.
The same matrix has a separate `native_runtime_smokes` section for framework-native flows such as
OpenAI Agents approval/resume, Strands AgentCore deployment, and Google ADK eval/deployment.
The matrix is intentionally readiness-only today; live provider/runtime calls should be added per
backend as small explicit smoke tests once the corresponding adapter path is stable enough for
networked CI.
