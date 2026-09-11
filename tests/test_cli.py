from __future__ import annotations

import json

from agentbridge.cli import main


def test_cli_lists_backends(capsys) -> None:
    assert main(["list-backends"]) == 0

    output = capsys.readouterr().out
    assert "mock" in output
    assert "langgraph" in output


def test_cli_inspects_backend_json(capsys) -> None:
    assert main(["inspect-backend", "mock", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "mock"
    assert payload["features"]["tools.sync"] == "full"


def test_cli_runs_manifest_with_mock_backend(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "agent.yaml"
    manifest_path.write_text(
        """
name: cli_agent
instructions: Reply to the user.
model: openai/gpt-5
tools:
  - name: lookup_order
""".strip()
    )

    assert (
        main(
            [
                "run",
                "--manifest",
                str(manifest_path),
                "--backend",
                "mock",
                "--input",
                "hello",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "mock"
    assert payload["output"]["agent"] == "cli_agent"
    assert payload["output"]["tools"][0]["name"] == "lookup_order"


def test_cli_compares_manifest_backends(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "agent.yaml"
    manifest_path.write_text(
        """
name: compare_agent
instructions: Compare backends.
model: openai/gpt-5
required_capabilities:
  - workflow.graph
  - tools.sync
""".strip()
    )

    assert (
        main(
            [
                "compare",
                "--manifest",
                str(manifest_path),
                "--backend",
                "mock",
                "--backend",
                "langgraph",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"] == "compare_agent"
    assert payload["backends"][0]["backend"] == "langgraph"
    assert payload["backends"][0]["features"]["workflow.graph"] == "full"


def test_cli_versions_json(capsys) -> None:
    assert main(["versions", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["pydantic_ai"]["package"] == "pydantic-ai-slim"
    assert payload["langgraph"]["adopted_range"] == ">=1.2.11,<2"
    assert payload["crewai"]["package"] == "agentbridge-crewai"
    assert payload["crewai"]["status"] == "blocked"


def test_cli_validates_manifest_json(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "agent.yaml"
    manifest_path.write_text(
        """
name: validate_agent
instructions: Validate me.
model: openai/gpt-5
tools:
  - name: lookup_order
required_capabilities:
  - workflow.graph
""".strip()
    )

    assert (
        main(
            [
                "validate",
                "--manifest",
                str(manifest_path),
                "--backend",
                "mock",
                "--backend",
                "langgraph",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"]
    assert payload["backend_fit"]["langgraph"] == "medium"


def test_cli_plugins_json(capsys) -> None:
    assert main(["plugins", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
