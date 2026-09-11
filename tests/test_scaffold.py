from __future__ import annotations

from agentbridge import AgentSpec, RunInput, get_adapter, list_adapters
from agentbridge.plugins import load_adapter_plugins, reset_plugin_loader
from agentbridge.scaffold import scaffold_adapter_plugin


def test_scaffold_adapter_plugin_creates_runnable_package(tmp_path, monkeypatch) -> None:
    target = tmp_path / "agentbridge-google-adk"

    created = scaffold_adapter_plugin(target, backend_name="google-adk")

    created_names = {item.path.relative_to(target).as_posix() for item in created}
    assert created_names == {
        "pyproject.toml",
        "README.md",
        "agentbridge_google_adk/__init__.py",
        "agentbridge_google_adk/adapter.py",
        "tests/test_adapter.py",
    }
    assert 'google_adk = "agentbridge_google_adk.adapter:Adapter"' in (
        target / "pyproject.toml"
    ).read_text()

    monkeypatch.syspath_prepend(str(target))
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGINS", "agentbridge_google_adk.adapter")
    reset_plugin_loader()

    results = load_adapter_plugins(force=True)

    assert any(result.loaded and result.backend == "google_adk" for result in results)
    assert "google_adk" in list_adapters()

    adapter = get_adapter("google_adk")
    spec = AgentSpec(name="demo", instructions="Echo.", model="mock/model")
    result = adapter.run(adapter.compile(spec), RunInput(input="hello"))
    assert result.backend == "google_adk"
    assert "hello" in result.output


def test_scaffold_adapter_plugin_refuses_invalid_backend_name(tmp_path) -> None:
    try:
        scaffold_adapter_plugin(tmp_path / "bad", backend_name="123-bad")
    except ValueError as exc:
        assert "backend_name" in str(exc)
    else:  # pragma: no cover - defensive assertion style
        raise AssertionError("Expected invalid backend name to fail")


def test_scaffold_adapter_plugin_does_not_overwrite_without_force(tmp_path) -> None:
    target = tmp_path / "plugin"
    scaffold_adapter_plugin(target, backend_name="custom")

    try:
        scaffold_adapter_plugin(target, backend_name="custom")
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover - defensive assertion style
        raise AssertionError("Expected existing files to fail without force")
