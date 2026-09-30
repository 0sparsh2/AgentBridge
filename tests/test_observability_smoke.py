from __future__ import annotations

from types import SimpleNamespace

from examples.observability_smoke import (
    run_langfuse_api_smoke_from_env,
    run_langfuse_runtime_smoke_from_env,
    run_langsmith_api_smoke_from_env,
    run_langsmith_prompt_smoke_from_env,
    run_observability_smoke_matrix_from_env,
)


def test_observability_smoke_matrix_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE", raising=False)

    matrix = run_observability_smoke_matrix_from_env()

    assert set(matrix) == {"langsmith_api", "langsmith_prompt", "langfuse_api", "langfuse_runtime"}
    assert all(item["status"] == "skipped" for item in matrix.values())


def test_langsmith_smoke_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    result = run_langsmith_api_smoke_from_env()

    assert result == {
        "ok": False,
        "status": "skipped",
        "missing_env": ["LANGSMITH_API_KEY"],
    }


def test_langsmith_smoke_calls_info_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def request_json(self, method, path):
            assert (method, path) == ("GET", "/info")
            return {"version": "test"}

    monkeypatch.setattr(
        "agentbridge_langchain.langsmith_api.LangSmithAPIClient",
        FakeClient,
    )

    result = run_langsmith_api_smoke_from_env()

    assert result == {"ok": True, "status": "verified", "response_keys": ["version"]}
    assert calls[0]["api_key"] == "test-key"


def test_langfuse_smoke_requires_both_credentials(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    result = run_langfuse_runtime_smoke_from_env()

    assert result["status"] == "skipped"
    assert result["missing_env"] == ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def test_langsmith_prompt_smoke_uses_native_prompt_bridge(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("AGENTBRIDGE_LANGSMITH_PROMPT", "refunds:latest")
    monkeypatch.setattr(
        "agentbridge_langchain.langsmith_prompts.pull_prompt",
        lambda identifier: {"identifier": identifier},
    )

    result = run_langsmith_prompt_smoke_from_env()

    assert result == {"ok": True, "status": "verified", "prompt_type": "dict"}


def test_langfuse_smoke_runs_offline_agent(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    fake_result = SimpleNamespace(output="offline response", backend="langchain")

    class FakeAdapter:
        def compile(self, agent):
            return agent

        def run(self, compiled, run_input):
            assert compiled.model == "agentbridge/offline"
            assert run_input.input == "hello"
            return fake_result

    monkeypatch.setattr("agentbridge_langchain.adapter.Adapter", FakeAdapter)

    result = run_langfuse_runtime_smoke_from_env()

    assert result == {
        "ok": True,
        "status": "verified",
        "backend": "langchain",
        "output_present": True,
    }


def test_langfuse_api_smoke_uses_native_transport(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["public_key"] == "pk"

        def request_json(self, method, path):
            assert (method, path) == ("GET", "/api/public/health")
            return {"status": "OK"}

    monkeypatch.setattr("agentbridge_langchain.langfuse_api.LangfuseAPIClient", FakeClient)
    result = run_langfuse_api_smoke_from_env()
    assert result == {"ok": True, "status": "verified", "response_keys": ["status"]}
