"""Scaffolding helpers for AgentBridge extension packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_BACKEND_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PACKAGE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ScaffoldedFile:
    """A file created by the plugin scaffolder."""

    path: Path
    created: bool


def scaffold_adapter_plugin(
    target_dir: str | Path,
    *,
    backend_name: str,
    package_name: str | None = None,
    distribution_name: str | None = None,
    force: bool = False,
) -> list[ScaffoldedFile]:
    """Create a minimal external adapter plugin package.

    The generated package is intentionally small and dependency-light. It is meant to be
    copied for new framework adapters, then filled in with native compile/run behavior.
    """

    backend_name = _validate_backend_name(backend_name)
    package_name = _validate_package_name(package_name or f"agentbridge_{backend_name}")
    distribution_name = distribution_name or package_name.replace("_", "-")

    root = Path(target_dir).expanduser().resolve()
    package_dir = root / package_name
    tests_dir = root / "tests"
    created: list[ScaffoldedFile] = []

    files = {
        root / "pyproject.toml": _pyproject_template(
            distribution_name=distribution_name,
            package_name=package_name,
            backend_name=backend_name,
        ),
        root / "README.md": _readme_template(
            distribution_name=distribution_name,
            package_name=package_name,
            backend_name=backend_name,
        ),
        package_dir / "__init__.py": _init_template(),
        package_dir / "adapter.py": _adapter_template(backend_name=backend_name),
        tests_dir / "test_adapter.py": _test_template(
            package_name=package_name,
            backend_name=backend_name,
        ),
    }

    for path, content in files.items():
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(ScaffoldedFile(path=path, created=True))

    return created


def _validate_backend_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not _BACKEND_NAME_PATTERN.match(normalized):
        raise ValueError(
            "backend_name must start with a letter and contain only lowercase letters, "
            "numbers, and underscores"
        )
    return normalized


def _validate_package_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not _PACKAGE_NAME_PATTERN.match(normalized):
        raise ValueError(
            "package_name must start with a letter and contain only lowercase letters, "
            "numbers, and underscores"
        )
    return normalized


def _pyproject_template(
    *,
    distribution_name: str,
    package_name: str,
    backend_name: str,
) -> str:
    return f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{distribution_name}"
version = "0.1.0"
description = "AgentBridge adapter plugin for {backend_name}."
readme = "README.md"
requires-python = ">=3.10"
license = {{ text = "MIT" }}
dependencies = [
  "agentbridge>=0.1.0",
]

[project.entry-points."agentbridge.adapters"]
{backend_name} = "{package_name}.adapter:Adapter"

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
"""


def _readme_template(
    *,
    distribution_name: str,
    package_name: str,
    backend_name: str,
) -> str:
    return f"""# {distribution_name}

AgentBridge adapter plugin for `{backend_name}`.

## Install

```bash
pip install -e .
```

## Verify Discovery

```bash
agentbridge plugins
agentbridge list-backends
agentbridge inspect-backend {backend_name} --json
agentbridge conformance --backend {backend_name}
```

## Local Development Without Installing

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="{package_name}.adapter"
agentbridge plugins
```

## Implementation Checklist

- Add the framework package dependency to `pyproject.toml`.
- Update `Adapter.capabilities()` with honest support metadata.
- Implement `compile()` by translating `AgentSpec` to native framework objects.
- Implement `run()` by returning a normalized `RunResult`.
- Implement `stream()` if the backend supports streaming.
- Add contract tests for every capability marked `full`.
- Run `agentbridge conformance --backend {backend_name}` before publishing.
- Document adopted and verified framework versions.
"""


def _init_template() -> str:
    return '''"""AgentBridge adapter plugin package."""

from .adapter import Adapter

__all__ = ["Adapter"]
'''


def _adapter_template(*, backend_name: str) -> str:
    return f'''"""AgentBridge adapter plugin for {backend_name}."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "{backend_name}"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={{
                "agent.instructions": "partial",
                "tools.sync": "unsupported",
                "streaming.events": "unsupported",
            }},
            notes={{
                "agent.instructions": "Template adapter echoes input until native integration is implemented.",
                "tools.sync": "Map ToolSpec to native tool definitions before marking this supported.",
            }},
        )

    def compile(self, spec: AgentSpec) -> AgentSpec:
        """Translate AgentSpec into native framework objects.

        Replace this with framework-specific construction. Returning the spec keeps
        the generated plugin runnable while the adapter is being developed.
        """

        return spec

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled agent and return a normalized result."""

        spec = compiled if isinstance(compiled, AgentSpec) else None
        output = f"[{{self.backend_name}}] {{run_input.input}}"
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=[
                AgentEvent(
                    type="message",
                    backend=self.backend_name,
                    data={{
                        "content": output,
                        "agent": spec.name if spec else None,
                    }},
                ),
                AgentEvent(
                    type="complete",
                    backend=self.backend_name,
                    data={{"output": output}},
                ),
            ],
            raw=compiled,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        result = self.run(compiled, run_input)
        yield from result.events
'''


def _test_template(*, package_name: str, backend_name: str) -> str:
    return f'''from __future__ import annotations

from {package_name}.adapter import Adapter
from agentbridge import AgentSpec, RunInput


def test_adapter_runs_template_agent() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="template_agent",
        instructions="Echo the user request.",
        model="mock/model",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "{backend_name}"
    assert "hello" in result.output
    assert [event.type for event in result.events] == ["message", "complete"]
'''
