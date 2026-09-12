from __future__ import annotations

import sys
from pathlib import Path

from agentbridge import get_adapter, list_adapters
from agentbridge.plugins import load_adapter_plugins, plugin_status, reset_plugin_loader


def test_env_plugin_registers_adapter(tmp_path, monkeypatch) -> None:
    plugin_path = tmp_path / "agentbridge_test_plugin.py"
    plugin_path.write_text(
        """
from agentbridge.adapters import BackendAdapter
from agentbridge.types import RunResult


class Adapter(BackendAdapter):
    backend_name = "env_plugin"

    def compile(self, spec):
        return spec

    def run(self, compiled, run_input):
        return RunResult(output="ok", backend=self.backend_name)
""".strip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGINS", "agentbridge_test_plugin")
    reset_plugin_loader()

    results = load_adapter_plugins(force=True)

    assert any(result.loaded and result.backend == "env_plugin" for result in results)
    assert "env_plugin" in list_adapters()
    assert get_adapter("env_plugin").backend_name == "env_plugin"


def test_plugin_status_reports_failed_env_plugin(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGINS", "missing_agentbridge_plugin")
    reset_plugin_loader()

    results = plugin_status()

    result = next(item for item in results if item.name == "missing_agentbridge_plugin")
    assert not result.loaded
    assert result.error


def test_env_plugin_cannot_override_builtin_without_opt_in(tmp_path, monkeypatch) -> None:
    plugin_path = tmp_path / "agentbridge_mock_plugin.py"
    plugin_path.write_text(
        """
from agentbridge.adapters import BackendAdapter


class Adapter(BackendAdapter):
    backend_name = "mock"
""".strip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGINS", "agentbridge_mock_plugin")
    monkeypatch.delenv("AGENTBRIDGE_ADAPTER_PLUGIN_OVERRIDES", raising=False)
    reset_plugin_loader()

    results = load_adapter_plugins(force=True)

    result = next(item for item in results if item.name == "agentbridge_mock_plugin")
    assert result.loaded is False
    assert "already registered" in result.error


def test_env_plugin_can_override_with_explicit_opt_in(tmp_path, monkeypatch) -> None:
    plugin_path = tmp_path / "agentbridge_override_plugin.py"
    plugin_path.write_text(
        """
from agentbridge.adapters import BackendAdapter


class Adapter(BackendAdapter):
    backend_name = "mock"
""".strip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGINS", "agentbridge_override_plugin")
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGIN_OVERRIDES", "mock")
    reset_plugin_loader()

    results = load_adapter_plugins(force=True)

    result = next(item for item in results if item.name == "agentbridge_override_plugin")
    assert result.loaded is True
    assert result.replaced is True
