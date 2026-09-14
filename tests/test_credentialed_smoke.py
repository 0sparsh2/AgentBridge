from __future__ import annotations

from examples.credentialed_smoke_matrix import build_smoke_matrix


def test_credentialed_smoke_matrix_is_disabled_by_default() -> None:
    matrix = build_smoke_matrix(environ={})

    assert matrix["enabled"] is False
    assert all(route["status"] == "skipped" for route in matrix["routes"].values())
    assert all(route["ready"] is False for route in matrix["routes"].values())
    assert all(
        smoke["status"] == "skipped"
        for smoke in matrix["native_runtime_smokes"].values()
    )


def test_credentialed_smoke_matrix_requires_route_specific_credentials() -> None:
    matrix = build_smoke_matrix(environ={"AGENTBRIDGE_RUN_CREDENTIAL_SMOKE": "1"})

    assert matrix["enabled"] is True
    assert matrix["routes"]["openai_api"]["missing_env"] == ["OPENAI_API_KEY"]
    assert matrix["routes"]["openrouter"]["missing_env"] == ["OPENROUTER_API_KEY"]
    assert matrix["routes"]["nvidia_nim_openai_compatible"]["missing_env"] == [
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_NIM_API_BASE or NVIDIA_NIM_BASE_URL",
    ]
    assert matrix["native_runtime_smokes"]["strands_agentcore_deployment"]["missing_env"] == [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AGENTBRIDGE_STRANDS_AGENTCORE_ROLE_ARN",
    ]
    assert matrix["native_runtime_smokes"]["google_adk_eval_deployment"]["missing_env"] == [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]


def test_credentialed_smoke_matrix_marks_ready_only_with_double_opt_in() -> None:
    matrix = build_smoke_matrix(
        environ={
            "AGENTBRIDGE_RUN_CREDENTIAL_SMOKE": "1",
            "OPENROUTER_API_KEY": "test-key",
        }
    )

    assert matrix["routes"]["openrouter"]["ready"] is True
    assert matrix["routes"]["openrouter"]["status"] == "ready"
    assert matrix["routes"]["openai_api"]["ready"] is False


def test_credentialed_smoke_matrix_accepts_nvidia_base_url_alias() -> None:
    matrix = build_smoke_matrix(
        environ={
            "AGENTBRIDGE_RUN_CREDENTIAL_SMOKE": "1",
            "NVIDIA_NIM_API_KEY": "test-key",
            "NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
            "NVIDIA_MODEL": "deepseek-ai/deepseek-v4-flash",
        }
    )

    route = matrix["routes"]["nvidia_nim_openai_compatible"]
    assert route["ready"] is True
    assert route["missing_env"] == []


def test_native_runtime_smokes_mark_ready_only_with_framework_credentials() -> None:
    matrix = build_smoke_matrix(
        environ={
            "AGENTBRIDGE_RUN_CREDENTIAL_SMOKE": "1",
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_REGION": "us-east-1",
            "AGENTBRIDGE_STRANDS_AGENTCORE_ROLE_ARN": "arn:aws:iam::123:role/test",
        }
    )

    assert matrix["native_runtime_smokes"]["strands_agentcore_deployment"]["ready"] is True
    assert matrix["native_runtime_smokes"]["openai_agents_native_resume"]["ready"] is False
