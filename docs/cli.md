# CLI Reference

The `agentbridge` CLI is intended for fast local validation, migration checks, examples, and smoke tests. It is not a hosted gateway.

## Install

```bash
pip install -e ".[dev]"
```

Optional framework adapters can be installed with extras:

```bash
pip install -e ".[langgraph]"
pip install -e ".[pydantic-ai]"
```

## Commands

### `list-backends`

List registered backends.

```bash
agentbridge list-backends
agentbridge list-backends --json
```

Use this after installing an adapter plugin to confirm discovery worked.

### `inspect-backend`

Inspect a backend's advertised capabilities.

```bash
agentbridge inspect-backend langgraph --json
```

This is the preferred way to check whether a backend is appropriate before trying to migrate a manifest.

### `run`

Run an agent manifest against one backend.

```bash
agentbridge run \
  --manifest examples/refund_agent.yaml \
  --backend mock \
  --tool-registry my_app.tools:build_registry \
  --input "Customer was double charged" \
  --json
```

The CLI resolves manifest tool names through an explicit registry. The built-in CLI registry only contains deterministic demo tools. Production apps should pass `--tool-registry module:attribute`.

### `compare`

Run the same manifest against multiple backends and compare normalized result shapes.

```bash
agentbridge compare \
  --manifest examples/refund_agent.yaml \
  --backend mock \
  --backend langgraph \
  --input "Customer was double charged" \
  --json
```

This supports the migration wedge: teams can measure what changes when moving a spec between runtimes.

### `validate`

Validate manifest structure, tool registry coverage, and selected backend capabilities.

```bash
agentbridge validate \
  --manifest examples/refund_agent.yaml \
  --backend mock \
  --backend langgraph \
  --tool-registry my_app.tools:build_registry \
  --json
```

Validation should run before adding a backend to a production migration path.

Tool registry references must use `module:attribute`. The attribute may be a `ToolRegistry`, a callable returning a `ToolRegistry`, or a dict/list/tuple of callables or `ToolSpec` objects.

### `capability-matrix`

Render backend support across the canonical AgentBridge capability taxonomy.

```bash
agentbridge capability-matrix --markdown
agentbridge capability-matrix --backend mock --backend langgraph --json
```

Use this as the fast coverage report for adapters and plugins. `--include-unknown` also displays adapter-reported features that are not yet part of the canonical taxonomy.

### `conformance`

Run lightweight adapter conformance checks.

```bash
agentbridge conformance --backend mock
agentbridge conformance --backend pydantic_ai --json
```

The conformance runner checks capability metadata, basic execution, streaming, sync tools when advertised as `full`, and structured output when advertised as `full`. This is not a replacement for backend-specific tests, but it is the baseline every adapter should pass before claiming compatibility.

### `extensions`

List framework-specific extension namespaces and config schemas.

```bash
agentbridge extensions
agentbridge extensions langgraph --json
```

Use this when you need to know which framework-specific knobs AgentBridge exposes outside the portable `AgentSpec` core.

### `versions`

Report adopted and installed package versions.

```bash
agentbridge versions --json
```

Keep the output aligned with [version_policy.md](version_policy.md).

### `plugins`

Inspect adapter plugin loading.

```bash
agentbridge plugins
agentbridge plugins --json
```

This reports loaded entry points, environment plugins, and plugin errors.

### `scaffold-plugin`

Create a starter external adapter plugin package.

```bash
agentbridge scaffold-plugin plugins/agentbridge-google-adk --backend google_adk
```

The generated package includes:

- `pyproject.toml` with an `agentbridge.adapters` entry point.
- A Python package containing `Adapter`.
- A README with install and verification commands.
- A starter test that runs the generated adapter.

Optional naming controls:

```bash
agentbridge scaffold-plugin ./my-plugin \
  --backend google_adk \
  --package agentbridge_google_adk \
  --distribution agentbridge-google-adk
```

Use `--force` only when intentionally regenerating files.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `AGENTBRIDGE_ADAPTER_PLUGINS` | Comma-separated module names to load as local adapter plugins. |
| `AGENTBRIDGE_ADAPTER_PLUGIN_OVERRIDES` | Comma-separated backend names a plugin is allowed to replace. |

## Exit Expectations

CLI commands should fail with clear, actionable messages when:

- A selected backend is unknown.
- An optional adapter dependency is missing.
- A manifest references a tool that is not registered.
- A backend does not meet required capabilities.
- A plugin import or registration fails.
