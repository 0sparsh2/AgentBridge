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


def test_cli_runs_manifest_with_explicit_tool_registry(tmp_path, capsys, monkeypatch) -> None:
    manifest_path = tmp_path / "agent.yaml"
    manifest_path.write_text(
        """
name: cli_agent
instructions: Reply to the user.
model: openai/gpt-5
tools:
  - name: lookup_customer
""".strip()
    )
    tools_path = tmp_path / "customer_tools.py"
    tools_path.write_text(
        """
def lookup_customer(customer_id: str) -> str:
    \"\"\"Look up a customer.\"\"\"
    return f"customer:{customer_id}"


TOOLS = [lookup_customer]
""".strip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert (
        main(
            [
                "run",
                "--manifest",
                str(manifest_path),
                "--backend",
                "mock",
                "--input",
                "C123",
                "--tool-registry",
                "customer_tools:TOOLS",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["output"]["tools"][0]["name"] == "lookup_customer"
    assert payload["output"]["tools"][0]["result"] == "customer:C123"


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


def test_cli_compare_adds_structured_output_requirement(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "agent.yaml"
    manifest_path.write_text(
        """
name: typed_agent
instructions: Return structured output.
model: openai/gpt-5
output_schema:
  type: object
  properties:
    answer:
      type: string
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
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert "structured_output" in payload["required_features"]
    assert payload["backends"][0]["features"]["structured_output"] == "unsupported"


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


def test_cli_capability_matrix_json(capsys) -> None:
    assert main(["capability-matrix", "--backend", "mock", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backends"] == ["mock"]
    feature_keys = [row["feature"]["key"] for row in payload["rows"]]
    assert "agent.instructions" in feature_keys
    assert "workflow.graph" in feature_keys


def test_cli_capability_matrix_markdown(capsys) -> None:
    assert main(["capability-matrix", "--backend", "mock", "--markdown"]) == 0

    output = capsys.readouterr().out
    assert "| Feature | Category | mock |" in output
    assert "`agent.instructions`" in output


def test_cli_conformance_json(capsys) -> None:
    assert main(["conformance", "--backend", "mock", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["backend"] == "mock"
    assert payload[0]["passed"] is True
    assert payload[0]["checks"][0]["name"] == "capabilities"


def test_cli_extensions_json(capsys) -> None:
    assert main(["extensions", "crewai", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["framework"] == "crewai"
    assert "role" in payload[0]["config_schema"]["properties"]


def test_cli_scaffolds_plugin(tmp_path, capsys) -> None:
    target = tmp_path / "plugin"

    assert main(["scaffold-plugin", str(target), "--backend", "strands"]) == 0

    output = capsys.readouterr().out
    assert "agentbridge_strands/adapter.py" in output
    assert (target / "pyproject.toml").exists()
    assert (target / "agentbridge_strands" / "adapter.py").exists()
