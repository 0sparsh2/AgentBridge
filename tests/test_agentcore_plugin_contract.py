from pathlib import Path


def test_agentcore_plugin_has_runtime_and_service_contract_docs():
    root = Path(__file__).parents[1] / "plugins" / "agentbridge-agentcore"
    assert (root / "agentbridge_agentcore" / "client.py").exists()
    readme = (root / "README.md").read_text()
    for term in ("Runtime", "Memory", "Gateway", "Observability"):
        assert term in readme
