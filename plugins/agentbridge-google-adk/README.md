# agentbridge-google-adk

AgentBridge adapter plugin for `google_adk`.

## Adopted Framework Version

- Native package: `google-adk`
- Adopted range: `>=2.9,<3`
- Verified locally: `2.9.0`
- Status: partial native adapter

## Target Capabilities

- Agent and tool mapping.
- Structured output through ADK `output_schema` plus AgentBridge validation into typed
  `AgentSpec.output_type` when available.
- Runner execution with in-memory or caller-supplied sessions.
- Session propagation from `RunInput.session_id`.
- Native ADK Agent options through `GoogleADKExtension`, including descriptions, global/static
  instructions, schemas, generation config, mode, transfer controls, planners, code executors,
  retry/timeout settings, lifecycle callbacks, and sub-agents.
- Native Runner options through `GoogleADKExtension`, including session service, memory service,
  artifact service, credential service, plugins, plugin close timeout, and auto-session behavior.
- Evals and deployment metadata as extension-level summaries.

## Dependency Note

`google-adk==2.9.0` currently pins older OpenTelemetry packages than the Strands stack. If you are developing both native plugins locally, prefer isolated virtual environments until those dependency ranges converge.

## Install

```bash
pip install -e .
```

## Verify Discovery

```bash
agentbridge plugins
agentbridge list-backends
agentbridge inspect-backend google_adk --json
agentbridge conformance --backend google_adk
```

The conformance runner uses the plugin-only `agentbridge/offline` model string. That path still
compiles a Google ADK `Agent` and executes through the native `Runner`, but uses a tiny local ADK
`BaseLlm` implementation so contract checks do not require cloud credentials.

`GoogleADKExtension` forwards native ADK Agent/Runner options when supplied and reports
serializable `run_kwargs`, `extension_config`, and `extension_summary` metadata on `RunResult`.
Eval execution and deployment publishing are still metadata-only until AgentBridge has native
no-network tests for those flows.

## Local Development Without Installing

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="agentbridge_google_adk.adapter"
agentbridge plugins
```

## Implementation Checklist

- Add the framework package dependency to `pyproject.toml`.
- Update `Adapter.capabilities()` with honest support metadata.
- Implement `compile()` by translating `AgentSpec` to native framework objects.
- Implement `run()` by returning a normalized `RunResult`.
- Implement `stream()` if the backend supports streaming.
- Add contract tests for every capability marked `full`.
- Run `agentbridge conformance --backend google_adk` before publishing.
- Document adopted and verified framework versions.
