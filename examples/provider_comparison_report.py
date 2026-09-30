"""Build a redacted, comparable report from provider smoke results."""

from __future__ import annotations

from typing import Any


def build_provider_comparison_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Normalize route results for docs, CI artifacts, and migration reports."""

    rows = []
    for route, result in results.items():
        rows.append(
            {
                "route": route,
                "status": result.get("status", "unknown"),
                "ok": bool(result.get("ok", False)),
                "model": result.get("model"),
                "base_url": result.get("base_url"),
                "output_present": bool(result.get("output") or result.get("output_present")),
                "error": result.get("error"),
            }
        )
    return {
        "schema_version": "agentbridge.provider-comparison.v1",
        "routes": rows,
        "verified_count": sum(row["ok"] for row in rows),
        "route_count": len(rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build_provider_comparison_report({}), indent=2))
