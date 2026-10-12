"""Report upstream releases that are outside AgentBridge adopted ranges."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from agentbridge.frameworks import FrameworkManifest, load_framework_manifests
from agentbridge.versioning import ADOPTED_DEPENDENCIES

CORE_DEPENDENCY_NAMES = frozenset({"pydantic", "litellm"})


def _latest_version(package: str) -> str:
    with urlopen(f"https://pypi.org/pypi/{package}/json", timeout=20) as response:
        payload = json.load(response)
    return str(payload["info"]["version"])


def _version_targets(manifests: Iterable[FrameworkManifest]) -> list[dict[str, str]]:
    """Return core and manifest-derived PyPI version targets exactly once."""

    targets = [
        {
            "backend": name,
            "package": str(dependency["package"]),
            "adopted_range": str(dependency["range"]),
        }
        for name, dependency in ADOPTED_DEPENDENCIES.items()
        if name in CORE_DEPENDENCY_NAMES
    ]
    targets.extend(
        {
            "backend": manifest.id,
            "package": manifest.package or "",
            "adopted_range": manifest.adopted_range or "",
        }
        for manifest in manifests
        if manifest.monitor_pypi
    )
    return sorted(targets, key=lambda target: target["backend"])


def check_versions(
    *,
    manifests: Iterable[FrameworkManifest] | None = None,
    latest_version: Callable[[str], str] = _latest_version,
) -> list[dict[str, str | bool]]:
    """Check core and manifest-backed packages against their adopted ranges."""

    selected_manifests = list(manifests) if manifests is not None else load_framework_manifests()
    results: list[dict[str, str | bool]] = []
    for target in _version_targets(selected_manifests):
        package = target["package"]
        adopted_range = target["adopted_range"]
        try:
            latest = latest_version(package)
            in_range = Version(latest) in SpecifierSet(adopted_range)
            results.append(
                {
                    "backend": target["backend"],
                    "package": package,
                    "adopted_range": adopted_range,
                    "latest": latest,
                    "in_adopted_range": in_range,
                }
            )
        except (KeyError, TypeError, ValueError, URLError, TimeoutError) as exc:
            results.append(
                {
                    "backend": target["backend"],
                    "package": package,
                    "adopted_range": adopted_range,
                    "latest": "unavailable",
                    "in_adopted_range": False,
                    "error": str(exc),
                }
            )
    return results


def render_markdown(results: Iterable[dict[str, str | bool]]) -> str:
    rows = [
        "# AgentBridge Upstream Version Check",
        "",
        "A release outside the adopted range requires compatibility review; it is not automatically supported.",
        "",
        "| Backend | Package | Adopted range | Latest | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        status = "in range" if result["in_adopted_range"] else "review needed"
        if result.get("error"):
            status = f"lookup failed: {result['error']}"
        rows.append(
            f"| `{result['backend']}` | `{result['package']}` | `{result['adopted_range']}` "
            f"| `{result['latest']}` | {status} |"
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    results = check_versions()
    report = render_markdown(results)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    drifted = [result for result in results if not result["in_adopted_range"]]
    if args.fail_on_drift and drifted:
        print(f"{len(drifted)} upstream package(s) require compatibility review.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
