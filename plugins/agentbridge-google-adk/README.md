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
- Run diagnostics for ADK event counts/types, function call/response counts, transfer-to-agent
  events, session/service bindings, sub-agent counts, eval labels, and deployment target labels.
- Opt-in service behavior snapshots through `capture_service_snapshots=True`, covering session
  lookup/listing/user state, memory search, artifact keys, and artifact versions when those native
  service methods are available.
- Eval labels, eval runner bindings, and structured deployment metadata as extension-level summaries.

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
serializable `run_kwargs`, `extension_config`, `extension_summary`, and `run_diagnostics` metadata
on `RunResult`.

`run_diagnostics` includes event history counts, text/function-call/function-response counts,
transfer-to-agent targets, session identity, service type summaries, optional service snapshots,
extension-level eval labels, eval runner bindings, and structured deployment metadata. Eval
execution and deployment publishing are still metadata-only until AgentBridge has native no-network
tests for those flows.

Service snapshots are disabled by default so AgentBridge does not accidentally call remote ADK
services. Enable them when you own the service objects and want post-run diagnostics:

```python
agent = GoogleADKExtension.with_config(
    agent,
    session_service=my_session_service,
    memory_service=my_memory_service,
    artifact_service=my_artifact_service,
    capture_service_snapshots=True,
)
```

When enabled, the adapter uses native ADK-shaped methods such as `get_session_sync`,
`list_sessions_sync`, `get_user_state`, `search_memory`, `list_artifact_keys`, and `list_versions`
when those methods exist. Missing methods and service errors are reported as safe diagnostic
records instead of failing the agent run.

Deployment metadata can be recorded without publishing to Google infrastructure:

```python
from agentbridge import AgentSpec
from agentbridge.extensions.google_adk import GoogleADKExtension

agent = GoogleADKExtension.with_config(
    AgentSpec(
        name="support_agent",
        instructions="Help customers with refunds.",
        model="google/gemini",
    ),
    evals=["refund_quality_eval"],
    deployment_target="vertex_ai",
    deployment={
        "runtime": "adk",
        "entrypoint": "app:agent",
        "region": "us-central1",
    },
)
```

This metadata is preserved in `RunResult.metadata["extension_summary"]["deployment"]` and
`RunResult.metadata["run_diagnostics"]["extension"]["deployment"]`.

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
