from __future__ import annotations

from agentbridge.frameworks import FrameworkManifest
from scripts.check_upstream_versions import check_versions


def test_upstream_version_check_uses_manifest_targets_and_skips_non_pypi() -> None:
    manifests = [
        FrameworkManifest(
            id="example_framework",
            display_name="Example Framework",
            package="example-framework",
            adopted_range=">=1,<2",
            status="planned",
            documentation_url="https://example.test/docs",
            next_slice="Add an adapter.",
        ),
        FrameworkManifest(
            id="service_only",
            display_name="Service Only",
            package="agentbridge-service-only",
            adopted_range=">=0.1,<1",
            monitor_pypi=False,
            status="planned",
            documentation_url="https://example.test/service",
            next_slice="Add a service smoke test.",
        ),
    ]

    versions = {
        "pydantic": "2.13.5",
        "litellm": "1.100.1",
        "example-framework": "1.5.0",
    }
    results = check_versions(manifests=manifests, latest_version=versions.__getitem__)

    assert [result["backend"] for result in results] == [
        "example_framework",
        "litellm",
        "pydantic",
    ]
    assert all(result["in_adopted_range"] for result in results)
