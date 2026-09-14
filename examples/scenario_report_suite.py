"""Generate an aggregate no-key scenario report suite."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable

from examples.crewai_prototype_report import build_report as build_crewai_report
from examples.google_adk_enterprise_report import build_report as build_google_adk_report
from examples.openai_agents_approval_report import build_report as build_openai_agents_report
from examples.pydantic_validation_report import build_report as build_pydantic_report
from examples.rag_migration_report import build_report as build_rag_report
from examples.strands_agentcore_report import build_report as build_strands_report


REPORT_BUILDERS: dict[str, Callable[[], dict[str, object]]] = {
    "crewai_prototype": build_crewai_report,
    "google_adk_enterprise": build_google_adk_report,
    "openai_agents_approval": build_openai_agents_report,
    "pydantic_validation": build_pydantic_report,
    "rag_migration": build_rag_report,
    "strands_agentcore": build_strands_report,
}


def build_suite() -> dict[str, object]:
    reports = {name: builder() for name, builder in REPORT_BUILDERS.items()}
    source_frameworks = sorted(
        {str(report["source_framework"]) for report in reports.values()}
    )
    target_frameworks = sorted(
        {str(report["target_framework"]) for report in reports.values()}
    )
    return {
        "suite": "AgentBridge scenario report suite",
        "reports_count": len(reports),
        "source_frameworks": source_frameworks,
        "target_frameworks": target_frameworks,
        "backend_availability": _backend_availability(reports),
        "scenario_index": {
            name: {
                "scenario": report["scenario"],
                "source_framework": report["source_framework"],
                "target_framework": report["target_framework"],
                "model_routes": sorted(report["model_routes"]),
                "backends": sorted(report["offline_run_comparison"]),
            }
            for name, report in reports.items()
        },
        "reports": reports,
    }


def _backend_availability(reports: dict[str, dict[str, object]]) -> dict[str, object]:
    seen: Counter[str] = Counter()
    available: Counter[str] = Counter()
    for report in reports.values():
        comparison = report.get("offline_run_comparison", {})
        if not isinstance(comparison, dict):
            continue
        for backend, backend_report in comparison.items():
            seen[str(backend)] += 1
            if isinstance(backend_report, dict) and backend_report.get("available") is True:
                available[str(backend)] += 1
    return {
        backend: {
            "seen": seen[backend],
            "available": available[backend],
            "unavailable": seen[backend] - available[backend],
        }
        for backend in sorted(seen)
    }


if __name__ == "__main__":
    print(json.dumps(build_suite(), indent=2, sort_keys=True))
