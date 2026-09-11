# Release Workflow

AgentBridge is not production-stable yet, but the repository has a release path ready for future PyPI publishing.

## Local Release Checks

Run these before creating a release tag:

```bash
python -m pip install -e ".[dev,langgraph,pydantic-ai]"
python -m pytest -q
agentbridge versions --json
agentbridge capability-matrix --markdown
agentbridge conformance --backend mock
python -m build
python -m twine check dist/*
```

## Version Update

1. Update `version` in `pyproject.toml`.
2. Update release notes or changelog once one exists.
3. Confirm [docs/version_policy.md](version_policy.md) matches adopted dependency ranges.
4. Commit the version change.

## Tag Format

Use `vMAJOR.MINOR.PATCH` tags:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Pushing a `v*` tag triggers `.github/workflows/release.yml`.

## PyPI Trusted Publishing

The release workflow uses PyPI trusted publishing through `pypa/gh-action-pypi-publish`.

Before the first real release:

- Create the PyPI project or reserve the package name.
- Configure PyPI trusted publishing for this GitHub repository.
- Require the `pypi` GitHub environment for publish approvals if desired.
- Verify the built package metadata with `twine check`.

## Release Workflow Jobs

- `build`: builds source and wheel distributions and uploads them as artifacts.
- `publish`: downloads distributions and publishes to PyPI from the `pypi` environment.

## Current Policy

Until v1, releases should be marked alpha and should not claim production stability. Adapter support must remain tied to the capability matrix and conformance tests.
