"""Framework support manifests and generated reporting helpers.

The manifests are a declared compatibility contract.  They complement runtime
adapter metadata: a manifest says what AgentBridge intends to support and how
it is verified, while an adapter reports what is available in the active
Python environment.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agentbridge.capabilities import CANONICAL_CAPABILITIES


SUPPORT_STATUSES = frozenset(
    {"core", "full", "partial", "extension", "native_only", "unsupported"}
)
FRAMEWORK_STATUSES = frozenset({"verified", "partial", "blocked", "planned"})
VERIFICATION_LANES = frozenset({"offline", "optional_dependency", "credentialed"})
CANONICAL_CAPABILITY_KEYS = frozenset(feature.key for feature in CANONICAL_CAPABILITIES)
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "frameworks"


class UpstreamSource(BaseModel):
    """One authoritative source used for compatibility review."""

    label: str
    url: str


class FrameworkManifest(BaseModel):
    """Declarative support contract for one framework integration."""

    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    package: str | None = None
    adopted_range: str | None = None
    monitor_pypi: bool = True
    plugin: str | None = None
    status: str
    documentation_url: str
    upstream_sources: list[UpstreamSource] = Field(default_factory=list)
    verification_lanes: list[str] = Field(default_factory=list)
    capabilities: dict[str, str] = Field(default_factory=dict)
    native_escape_hatches: list[str] = Field(default_factory=list)
    next_slice: str

    @model_validator(mode="after")
    def validate_contract(self) -> "FrameworkManifest":
        if self.status not in FRAMEWORK_STATUSES:
            raise ValueError(f"status must be one of {sorted(FRAMEWORK_STATUSES)}")
        if (self.package is None) != (self.adopted_range is None):
            raise ValueError("package and adopted_range must either both be set or both be omitted")
        if self.monitor_pypi and self.package is None:
            raise ValueError("monitor_pypi requires package and adopted_range")
        unknown_lanes = sorted(set(self.verification_lanes) - VERIFICATION_LANES)
        if unknown_lanes:
            raise ValueError(f"unknown verification lane(s): {', '.join(unknown_lanes)}")
        if len(self.verification_lanes) != len(set(self.verification_lanes)):
            raise ValueError("verification_lanes must not contain duplicates")
        unknown_capabilities = sorted(set(self.capabilities) - CANONICAL_CAPABILITY_KEYS)
        if unknown_capabilities:
            raise ValueError(
                "capabilities must use the canonical taxonomy; unknown key(s): "
                + ", ".join(unknown_capabilities)
            )
        invalid_statuses = {
            key: value for key, value in self.capabilities.items() if value not in SUPPORT_STATUSES
        }
        if invalid_statuses:
            raise ValueError(f"invalid capability status(es): {invalid_statuses}")
        return self


class FrameworkSupportReport(BaseModel):
    """Stable, machine-readable summary of all declared framework support."""

    frameworks: list[FrameworkManifest]
    framework_statuses: dict[str, int]
    capability_statuses: dict[str, int]

    def as_markdown(self) -> str:
        """Render a generated support page suitable for project documentation."""

        lines = [
            "# Framework Support",
            "",
            "This page is generated from the versioned files in `frameworks/`. It records the "
            "declared support contract; runtime availability is still reported by "
            "`agentbridge coverage-report`.",
            "",
            "## Summary",
            "",
            "| Framework status | Count |",
            "| --- | ---: |",
        ]
        for status, count in sorted(self.framework_statuses.items()):
            lines.append(f"| `{status}` | {count} |")
        lines.extend(
            [
                "",
                "## Integrations",
                "",
                "| Framework | Package | Status | Verification lanes | Declared capabilities |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for framework in self.frameworks:
            package = "-"
            if framework.package and framework.adopted_range:
                package = f"`{framework.package}` `{framework.adopted_range}`"
            lanes = ", ".join(f"`{lane}`" for lane in framework.verification_lanes) or "-"
            lines.append(
                f"| [{framework.display_name}]({framework.documentation_url}) | {package} "
                f"| `{framework.status}` | {lanes} | {len(framework.capabilities)} |"
            )
        for framework in self.frameworks:
            lines.extend(
                [
                    "",
                    f"## {framework.display_name}",
                    "",
                    f"- Framework ID: `{framework.id}`",
                    f"- Plugin: `{framework.plugin or 'built-in or documentation-only'}`",
                    f"- Next slice: {framework.next_slice}",
                ]
            )
            if framework.native_escape_hatches:
                lines.append(
                    "- Native escape hatches: "
                    + ", ".join(f"`{item}`" for item in framework.native_escape_hatches)
                )
            if framework.upstream_sources:
                lines.append(
                    "- Upstream review sources: "
                    + ", ".join(f"[{source.label}]({source.url})" for source in framework.upstream_sources)
                )
            if framework.capabilities:
                lines.extend(["", "| Capability | Declared support |", "| --- | --- |"])
                for capability, status in sorted(framework.capabilities.items()):
                    lines.append(f"| `{capability}` | `{status}` |")
        lines.append("")
        return "\n".join(lines)


def load_framework_manifests(directory: Path | str | None = None) -> list[FrameworkManifest]:
    """Load and validate every versioned framework support manifest."""

    manifest_dir = Path(directory) if directory is not None else DEFAULT_MANIFEST_DIR
    manifests: list[FrameworkManifest] = []
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid framework manifest {path}: {exc}") from exc
        try:
            manifests.append(FrameworkManifest.model_validate(payload))
        except ValueError as exc:
            raise ValueError(f"Invalid framework manifest {path}: {exc}") from exc
    identifiers = [manifest.id for manifest in manifests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Framework manifest IDs must be unique")
    return manifests


def framework_support_report(directory: Path | str | None = None) -> FrameworkSupportReport:
    """Return the aggregate declared support contract."""

    manifests = load_framework_manifests(directory)
    framework_statuses = Counter(manifest.status for manifest in manifests)
    capability_statuses = Counter(
        status for manifest in manifests for status in manifest.capabilities.values()
    )
    return FrameworkSupportReport(
        frameworks=manifests,
        framework_statuses=dict(sorted(framework_statuses.items())),
        capability_statuses=dict(sorted(capability_statuses.items())),
    )
