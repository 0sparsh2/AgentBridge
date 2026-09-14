from __future__ import annotations

from examples.credentialed_smoke_matrix import build_smoke_matrix


def test_credentialed_smoke_matrix_is_disabled_by_default() -> None:
    matrix = build_smoke_matrix(environ={})

    assert matrix["enabled"] is False
    assert all(route["status"] == "skipped" for route in matrix["routes"].values())
    assert all(route["ready"] is False for route in matrix["routes"].values())


def test_credentialed_smoke_matrix_requires_route_specific_credentials() -> None:
    matrix = build_smoke_matrix(environ={"AGENTBRIDGE_RUN_CREDENTIAL_SMOKE": "1"})

    assert matrix["enabled"] is True
    assert matrix["routes"]["openai_api"]["missing_env"] == ["OPENAI_API_KEY"]
    assert matrix["routes"]["openrouter"]["missing_env"] == ["OPENROUTER_API_KEY"]
    assert matrix["routes"]["nvidia_nim_openai_compatible"]["missing_env"] == [
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_NIM_BASE_URL",
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
