from __future__ import annotations

from examples.live_model_smoke import (
    run_nvidia_nim_smoke_from_env,
    run_provider_smoke_matrix_from_env,
)


def test_nvidia_nim_live_smoke_is_skipped_without_global_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", raising=False)

    result = run_nvidia_nim_smoke_from_env()

    assert result["ok"] is False
    assert result["status"] == "skipped"


def test_nvidia_nim_live_smoke_accepts_user_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", "1")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")

    calls = {}

    def fake_smoke(**kwargs):
        calls.update(kwargs)
        return {"ok": True, "content": "AgentBridge smoke test ok"}

    monkeypatch.setattr("examples.live_model_smoke.run_openai_compatible_smoke", fake_smoke)

    result = run_nvidia_nim_smoke_from_env()

    assert result["ok"] is True
    assert calls["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert calls["model"] == "deepseek-ai/deepseek-v4-flash-0731"


def test_provider_smoke_matrix_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", raising=False)

    matrix = run_provider_smoke_matrix_from_env()

    assert set(matrix) == {"nvidia_nim", "openrouter", "ollama"}
    assert all(result["status"] == "skipped" for result in matrix.values())


def test_provider_smoke_matrix_routes_openrouter_and_ollama(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
    calls = []

    def fake_smoke(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "status": 200, "content": "ok"}

    monkeypatch.setattr("examples.live_model_smoke.run_openai_compatible_smoke", fake_smoke)
    matrix = run_provider_smoke_matrix_from_env()

    assert matrix["openrouter"]["ok"] is True
    assert matrix["ollama"]["ok"] is True
    assert matrix["nvidia_nim"]["status"] == "skipped"
    assert {call["base_url"] for call in calls} == {
        "https://openrouter.ai/api/v1",
        "http://ollama.test/v1",
    }


def test_provider_smoke_matrix_supports_nvidia_alias(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", "1")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_NIM_BASE_URL", "https://nim.test/v1")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/test")
    monkeypatch.setattr(
        "examples.live_model_smoke.run_openai_compatible_smoke",
        lambda **kwargs: {"ok": True, **kwargs},
    )

    result = run_provider_smoke_matrix_from_env()["nvidia_nim"]

    assert result["ok"] is True
    assert result["base_url"] == "https://nim.test/v1"
    assert result["model"] == "nvidia/test"
