from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from agentbridge.adapters import BackendAdapter


def test_crewai_plugin_scaffold_exposes_adapter_without_crewai_installed() -> None:
    adapter_path = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "agentbridge-crewai"
        / "agentbridge_crewai"
        / "adapter.py"
    )
    spec = importlib.util.spec_from_file_location("agentbridge_crewai_test_adapter", adapter_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    adapter_type = module.CrewAIAdapter

    assert issubclass(adapter_type, BackendAdapter)
    assert adapter_type.backend_name == "crewai"
    assert adapter_type().capabilities().status("workflow.roles_tasks") == "full"
