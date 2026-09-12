from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput, get_adapter, list_adapters
from agentbridge.plugins import load_adapter_plugins, reset_plugin_loader


ROOT = Path(__file__).resolve().parents[1]


class FakeOpenAIAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOpenAIResult:
    final_output = "hello from native openai agents"
    usage = {"requests": 1}
    new_items = [SimpleNamespace(content="hello from native openai agents")]


class FakeOpenAIRunner:
    @staticmethod
    def run_sync(agent, input, **kwargs):
        return FakeOpenAIResult()

    @staticmethod
    def run_streamed(agent, input, **kwargs):
        return FakeOpenAIResult()


def fake_openai_function_tool(func, **kwargs):
    return SimpleNamespace(func=func, kwargs=kwargs)


def install_fake_openai_agents(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "agents",
        SimpleNamespace(
            Agent=FakeOpenAIAgent,
            Runner=FakeOpenAIRunner,
            function_tool=fake_openai_function_tool,
        ),
    )


def test_next_wave_plugin_scaffolds_load_locally(monkeypatch) -> None:
    install_fake_openai_agents(monkeypatch)
    plugin_roots = [
        ROOT / "plugins" / "agentbridge-openai-agents",
        ROOT / "plugins" / "agentbridge-google-adk",
        ROOT / "plugins" / "agentbridge-strands",
        ROOT / "plugins" / "agentbridge-langchain",
    ]
    for root in plugin_roots:
        monkeypatch.syspath_prepend(str(root))

    monkeypatch.setenv(
        "AGENTBRIDGE_ADAPTER_PLUGINS",
        ",".join(
            [
                "agentbridge_openai_agents.adapter",
                "agentbridge_google_adk.adapter",
                "agentbridge_strands.adapter",
                "agentbridge_langchain.adapter",
            ]
        ),
    )
    reset_plugin_loader()

    results = load_adapter_plugins(force=True)
    loaded = {result.backend for result in results if result.loaded}

    assert {"openai_agents", "google_adk", "strands", "langchain"}.issubset(loaded)
    assert {"openai_agents", "google_adk", "strands", "langchain"}.issubset(list_adapters())

    spec = AgentSpec(name="demo", instructions="Echo input.", model="mock/model")
    for backend in ["openai_agents", "google_adk", "strands", "langchain"]:
        adapter = get_adapter(backend)
        result = adapter.run(adapter.compile(spec), RunInput(input="hello"))
        assert result.backend == backend
        assert "hello" in result.output


def test_next_wave_plugin_capabilities_are_honest(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "plugins" / "agentbridge-strands"))
    monkeypatch.setenv("AGENTBRIDGE_ADAPTER_PLUGINS", "agentbridge_strands.adapter")
    reset_plugin_loader()
    load_adapter_plugins(force=True)

    capabilities = get_adapter("strands").capabilities()

    assert capabilities.status("tools.sync") == "unsupported"
    assert capabilities.status("tools.mcp") == "extension"
    assert capabilities.status("observability.tracing") == "extension"
