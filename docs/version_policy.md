# Version Policy

AgentBridge adapters must document the framework versions they target. Agent frameworks move quickly, so every adapter has two version concepts:

- Adopted baseline: the minimum version AgentBridge intentionally targets in `pyproject.toml`.
- Verified version: the exact version exercised in local or CI tests.

## Current Adopted Baselines

| Package | Purpose | Adopted Range | Verified Locally | Status |
| --- | --- | --- | --- | --- |
| `pydantic` | Core SDK models | `>=2.13,<3` | `2.13.5` | Verified |
| `litellm` | Model-provider delegation | `>=1.100,<2` | `1.100.1` | Verified install |
| `langgraph` | Stateful graph backend | `>=1.2.11,<2` | `1.2.11` | Verified graph + tool execution |
| `pydantic-ai-slim` | Typed-agent backend import package | `>=2.42,<3` | `2.42.0` | Verified with offline `TestModel` |
| `agentbridge-crewai` | External CrewAI adapter plugin | upstream `crewai>=0.11.2,<0.12` | Scaffolded, not installed | Blocked on Python 3.14 dependency resolution |
| `openai-agents` | External OpenAI Agents SDK adapter plugin | `>=0.20,<0.21` | Latest observed `0.22.2`; compatible baseline `0.20.0` | Partial native adapter |
| `google-adk` | External Google ADK adapter plugin | `>=2.9,<3` | Latest observed `2.9.0` | Planned plugin scaffold |
| `strands-agents` | External Strands Agents adapter plugin | `>=1.55,<2` | `1.55.1` | Partial native adapter |
| `langchain` | External direct LangChain adapter plugin | `>=1.4,<2` | `1.4.0` | Partial native adapter |

## Notes

- `langgraph` is the first real framework adapter verified locally because the current adapter can execute without model credentials.
- `pydantic-ai-slim` is preferred over the full `pydantic-ai` meta-package because AgentBridge adapters should not install every provider/eval/MCP extra just to expose the core `pydantic_ai` module.
- CrewAI `0.11.2` failed dependency resolution in this Python 3.14 environment because it depends on older LangChain/LangSmith ranges. Keep the adapter as a skeleton until we either test it in an older Python environment or adopt a newer CrewAI package from an index that resolves cleanly.
- OpenAI Agents, Google ADK, Strands, and direct LangChain support are intentionally external plugin targets. Their native packages are not part of the core `agentbridge` install path.
- OpenAI Agents `0.22.x` requires `openai>=3`, while the current AgentBridge core dependency path through LiteLLM uses `openai<3`. The OpenAI Agents plugin therefore adopts `0.20.x` for executable work first and treats `0.22.x` as blocked until the major-version conflict is resolved.
- Latest observed versions for planned plugins were checked with `pip index versions` on 2026-09-11. They are not verified adapter execution versions yet.
- If upstream APIs break within a range, tighten the range and update this document in the same change.
- Never claim full adapter support for a framework version unless a contract test runs against that exact version.

## Future Framework Matrix

Before adding a new framework adapter, document:

- Package name and install extra.
- Adopted version range.
- Verified version.
- Core features supported through `AgentSpec`.
- Features exposed through capability metadata.
- Native-only features that require raw backend access.

## CLI Verification

Use this command to inspect adopted and installed dependency versions:

```bash
agentbridge versions --json
```
