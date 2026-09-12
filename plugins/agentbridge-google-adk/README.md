# agentbridge-google-adk

AgentBridge adapter plugin for `google_adk`.

## Adopted Framework Version

- Native package: `google-adk`
- Adopted range: `>=2.9,<3`
- Verified locally: `2.9.0`
- Status: partial native adapter

## Target Capabilities

- Agent and tool mapping.
- Runner execution with in-memory sessions.
- Session propagation from `RunInput.session_id`.
- Memory and artifact service configuration through `GoogleADKExtension`.
- Sub-agent/delegation metadata.
- Evals and deployment metadata as extension-level features.

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
