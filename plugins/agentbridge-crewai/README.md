# AgentBridge CrewAI Adapter Plugin

This package is the external CrewAI adapter for AgentBridge.

CrewAI is intentionally kept outside the core `agentbridge` package because the currently adopted package range has a heavy and unresolved dependency tree in the Python 3.14 development environment.

## Install

```bash
pip install -e plugins/agentbridge-crewai
```

## Status

Blocked in the current workspace:

- Adopted range: `crewai>=0.11.2,<0.12`
- Issue: dependency resolution conflict involving older LangChain/LangSmith ranges on Python 3.14

The adapter code is scaffolded so it can be verified in a compatible Python environment without forcing the core SDK to install CrewAI.
