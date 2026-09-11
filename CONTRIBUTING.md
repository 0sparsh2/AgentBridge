# Contributing

## Development Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,langgraph,pydantic-ai]"
.venv/bin/python -m pytest -q
```

## Git Workflow

- Keep `main` deployable and green.
- Use focused branches for changes.
- Keep commits small and descriptive.
- Open issues for meaningful follow-up work.
- Update tests and docs with behavior changes.

## Adapter Rules

- Do not add heavy framework dependencies to core without a clear reason.
- Prefer external plugins for frameworks with large or conflicting dependency trees.
- Any adapter capability marked `full` should have a contract test.
- Update `docs/version_policy.md` when dependency ranges or verification status changes.
