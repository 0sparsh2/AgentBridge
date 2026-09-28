"""Report upstream releases that are outside AgentBridge adopted ranges."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from agentbridge.versioning import ADOPTED_DEPENDENCIES

UPSTREAM_PACKAGE_NAMES = {
    "crewai": "crewai",
}


def _latest_version(package: str) -> str:
    with urlopen(f"https://pypi.org/pypi/{package}/json", timeout=20) as response:
        payload = json.load(response)
    return str(payload["info"]["version"])


def check_versions() -> list[dict[str, str | bool]]:
    results: list[dict[str, str | bool]] = []
    for name, dependency in ADOPTED_DEPENDENCIES.items():
        package = UPSTREAM_PACKAGE_NAMES.get(name, str(dependency["package"]))
        adopted_range = str(dependency["range"])
        try:
            latest = _latest_version(package)
            in_range = Version(latest) in SpecifierSet(adopted_range)
            results.append(
                {
                    "backend": name,
                    "package": package,
                    "adopted_range": adopted_range,
                    "latest": latest,
                    "in_adopted_range": in_range,
                }
            )
        except (KeyError, TypeError, ValueError, URLError, TimeoutError) as exc:
            results.append(
                {
                    "backend": name,
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
