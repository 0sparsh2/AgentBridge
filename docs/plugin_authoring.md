# Adapter Plugin Authoring

AgentBridge keeps heavy or blocked framework adapters outside the core install path. External packages can register adapters through Python entry points, while local development can use an environment variable.

## Generate A Starter Plugin

Use the CLI scaffolder for a new adapter package:

```bash
agentbridge scaffold-plugin plugins/agentbridge-google-adk --backend google_adk
```

The generated package includes:

- `pyproject.toml` with an `agentbridge.adapters` entry point.
- A package containing an `Adapter` class.
- A README with install and verification commands.
- A starter test for the generated adapter.

Install and verify:

```bash
cd plugins/agentbridge-google-adk
pip install -e ".[dev]"
pytest
agentbridge plugins
agentbridge inspect-backend google_adk --json
```

The generated adapter intentionally echoes input. Replace `compile()`, `run()`, and `stream()` with native framework behavior before marking real capabilities as supported.

## Entry Point Plugins

External packages should expose a `BackendAdapter` subclass:

```python
from agentbridge.adapters import BackendAdapter


class MyFrameworkAdapter(BackendAdapter):
    backend_name = "my_framework"
```

Then register it in `pyproject.toml`:

```toml
[project.entry-points."agentbridge.adapters"]
my_framework = "my_package.adapter:MyFrameworkAdapter"
```

AgentBridge discovers the adapter automatically when `list_adapters()`, `get_adapter()`, or CLI commands access the registry.

## Local Development Plugins

Set `AGENTBRIDGE_ADAPTER_PLUGINS` to comma-separated module names:

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="my_local_adapter,another_adapter"
agentbridge plugins
```

Each module must expose either:

- `Adapter`: a `BackendAdapter` subclass.
- `get_adapter()`: returns a `BackendAdapter` subclass or instance.

## Status and Debugging

Use:

```bash
agentbridge plugins --json
```

The command reports which plugins loaded, which backend names they registered, and any import or validation errors.

## Override Safety

Plugins cannot overwrite an existing backend by default. To intentionally replace a backend during development:

```bash
export AGENTBRIDGE_ADAPTER_PLUGIN_OVERRIDES="mock"
export AGENTBRIDGE_ADAPTER_PLUGINS="my_mock_replacement"
```

Use overrides sparingly. Published adapter packages should avoid backend names owned by core AgentBridge unless the package is intentionally replacing that adapter.

During tests or interactive development, `reset_adapters()` restores the registry to the built-in adapters, and `reset_plugin_loader()` also clears plugin discovery state.

## CrewAI Recommendation

CrewAI is currently blocked in the core Python 3.14 environment because its adopted package range pulls older LangChain/LangSmith dependencies. The scaffold lives in `plugins/agentbridge-crewai` as an external plugin package with its own Python and dependency constraints.
