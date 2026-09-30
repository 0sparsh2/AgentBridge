from examples.provider_comparison_report import build_provider_comparison_report


def test_provider_comparison_report_normalizes_routes_without_secrets():
    report = build_provider_comparison_report(
        {
            "nvidia_nim": {
                "ok": True,
                "status": "verified",
                "model": "deepseek-ai/deepseek-v4-flash",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "output": "answer",
                "api_key": "must-not-be-copied",
            },
            "ollama": {"ok": False, "status": "failed", "error": "ConnectionError"},
        }
    )
    assert report["schema_version"] == "agentbridge.provider-comparison.v1"
    assert report["verified_count"] == 1
    assert report["routes"][0]["output_present"] is True
    assert "api_key" not in report["routes"][0]
