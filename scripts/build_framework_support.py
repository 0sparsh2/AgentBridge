"""Generate and validate the framework support documentation from manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentbridge.frameworks import framework_support_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "framework_support.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if generated output is stale.")
    args = parser.parse_args()

    rendered = framework_support_report().as_markdown()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            print(f"Generated framework support is stale: {args.output}", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
