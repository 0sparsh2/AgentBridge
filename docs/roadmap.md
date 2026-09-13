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

- Pydantic AI adapter execution with tool calling.
- LangGraph minimal graph execution, conditional routing, checkpoint interrupt reporting, and compiled-runtime resume.
- Contract tests gated behind optional dependency markers.
- CLI commands for running, validating, comparing, inspecting, and conformance-checking backends.
- Backend capability metadata for supported features and known gaps.
- `docs/version_policy.md` kept current as adapter dependency ranges change.
- Static JSON/YAML manifests for CLI runs.
- Scaffold command for lightweight external adapter plugins.
- External plugins for OpenAI Agents SDK, Strands Agents, direct LangChain, and Google ADK with adopted version ranges, offline contract tests, and capability-specific diagnostics.
- CrewAI dependency strategy remains unresolved in this Python 3.14 environment and should be verified separately before claiming native execution.

## v1: Migration and Comparison

- Agent manifest file format.
- `agentbridge compare` command for backend fit analysis.
- Tool registry for manifest-declared tools.
- Existing-project import helpers for common CrewAI and LangGraph structures.
- Existing-project import helpers for common LangChain agent apps.
- Conservative `MigrationReport` helpers for LangChain and LangGraph object shapes.
- Tracing and diagnostics surfaces for plugin backends, with provider-specific smoke tests where credentials/config are available.
- Full AG-UI server example.

## v2: Full Framework Coverage

- Capability matrix for each supported framework.
- Framework-specific extension modules for features that do not fit the common core.
- Conformance test suite that every adapter must pass.
- Coverage reports showing which framework features are fully supported, partially supported, unsupported, or intentionally native-only.
- Adapter authoring guide for new frameworks and tools.
- First-class external adapter packages for OpenAI Agents SDK, Strands Agents, LangChain, Google ADK, CrewAI, and additional ecosystem frameworks.
