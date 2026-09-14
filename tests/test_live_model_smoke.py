from __future__ import annotations

from examples.live_model_smoke import run_nvidia_nim_smoke_from_env


def test_nvidia_nim_live_smoke_is_skipped_without_global_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", raising=False)

    result = run_nvidia_nim_smoke_from_env()

    assert result["ok"] is False
    assert result["status"] == "skipped"


def test_nvidia_nim_live_smoke_accepts_user_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", "1")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")

    calls = {}

    def fake_smoke(**kwargs):
        calls.update(kwargs)
        return {"ok": True, "content": "AgentBridge smoke test ok"}

    monkeypatch.setattr("examples.live_model_smoke.run_openai_compatible_smoke", fake_smoke)

    result = run_nvidia_nim_smoke_from_env()

    assert result["ok"] is True
    assert calls["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert calls["model"] == "deepseek-ai/deepseek-v4-flash"
