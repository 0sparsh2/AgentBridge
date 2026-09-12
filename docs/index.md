# AgentBridge Documentation

Use this page as the navigation map for the repository.

## Start Here

- [README](../README.md): Public project overview, quickstarts, architecture summary, and development workflow.
- [Vision And Target State](vision.md): Final target, goals, plugin families, and long-term ecosystem map.
- [Requirements](requirements.md): Product requirements, target users, non-goals, success criteria, and acceptance criteria.
- [Roadmap](roadmap.md): v0, v0.1, and v1 milestones.

## Product And Research

- [Research](research.md): Comparison of AG-UI, LiteLLM, CrewAI, LangGraph, and Pydantic AI.
- [Adapter Target Research](adapter_target_research.md): Next adapter targets including OpenAI Agents SDK, Strands, direct LangChain, and Google ADK.
- [Capability Coverage](capability_coverage.md): Long-term strategy for covering framework nuance without flattening everything into a weak abstraction.
- [Version Policy](version_policy.md): Adopted package ranges, verified local versions, and adapter version rules.
- [Release Workflow](release.md): Package build checks, tag workflow, and future PyPI publishing path.

## Engineering

- [Architecture](architecture.md): System architecture, data flow, plugin loading, manifest flow, and normalized type model.
- [Design](design.md): Design decisions, tradeoffs, extension principles, and boundaries.
- [Adapters](adapters.md): Backend support matrix, current limitations, and adapter-specific notes.
- [CLI](cli.md): Command reference for validating, running, comparing, and inspecting agents.
- [Conformance](conformance.md): Adapter contract checks and offline plugin model rules.
- [Migration Helpers](migration.md): Conservative import reports for LangChain and LangGraph project shapes.
- [Plugin Authoring](plugin_authoring.md): How to build adapter plugins outside the core package.

## Examples

- [examples/refund_agent.py](../examples/refund_agent.py): Same agent spec against selectable framework adapters.
- [examples/research_agent.py](../examples/research_agent.py): Tool use and streaming event demo.
- [examples/agui_events.py](../examples/agui_events.py): Convert normalized AgentBridge events into AG-UI-shaped events.
- [examples/refund_agent.yaml](../examples/refund_agent.yaml): Static manifest for CLI runs.

## Maintainer Checklist

Before claiming support for a framework feature:

- Add or update adapter capability metadata.
- Add contract tests for every capability marked `full`.
- Run conformance for the changed backend.
- Update [Version Policy](version_policy.md) with adopted and verified versions.
- Update [Adapters](adapters.md) with limitations and native escape hatches.
- Add example coverage when the feature changes user-facing behavior.
