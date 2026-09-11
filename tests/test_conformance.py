from __future__ import annotations

from agentbridge import run_conformance


def test_run_conformance_passes_for_mock_backend() -> None:
    reports = run_conformance(backends=["mock"])

    assert len(reports) == 1
    report = reports[0]
    assert report.backend == "mock"
    assert report.passed
    checks = {check.name: check for check in report.checks}
    assert checks["capabilities"].passed
    assert checks["basic_run"].passed
    assert checks["stream"].passed
    assert checks["sync_tools"].passed
    assert checks["structured_output"].skipped


def test_run_conformance_as_dict_includes_passed_status() -> None:
    report = run_conformance(backends=["mock"])[0]

    payload = report.as_dict()

    assert payload["backend"] == "mock"
    assert payload["passed"] is True
