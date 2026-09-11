# AgentBridge Roadmap

## v0: SDK Foundation

- Requirements, research, architecture, and roadmap docs.
- Python package scaffold.
- Core framework-neutral types.
- Adapter registry.
- Mock backend.
- Optional adapter skeletons for Pydantic AI, CrewAI, and LangGraph.
- Examples for refunds, research, and AG-UI-shaped events.
- Unit tests without paid API calls.

## v0.1: Real Backend Execution

- Complete Pydantic AI adapter execution with tool calling.
- Resolve CrewAI dependency strategy before completing role/task/crew execution.
- Complete LangGraph adapter execution for a minimal state graph.
- Add contract tests gated behind optional dependency markers.
- Add CLI command for running an example agent against a selected backend.
- Add backend capability metadata for supported features and known gaps.
- Keep `docs/version_policy.md` updated as adapter dependency ranges change.
- Add static JSON/YAML manifests for CLI runs.

## v1: Migration and Comparison

- Agent manifest file format.
- `agentbridge compare` command for backend fit analysis.
- Tool registry for manifest-declared tools.
- Existing-project import helpers for common CrewAI and LangGraph structures.
- Tracing integrations.
- Full AG-UI server example.

## v2: Full Framework Coverage

- Capability matrix for each supported framework.
- Framework-specific extension modules for features that do not fit the common core.
- Conformance test suite that every adapter must pass.
- Coverage reports showing which framework features are fully supported, partially supported, unsupported, or intentionally native-only.
- Adapter authoring guide for new frameworks and tools.
