"""AgentBridge public API."""

from agentbridge.agui import event_to_agui
from agentbridge.capabilities import (
    CANONICAL_CAPABILITIES,
    CapabilityFeature,
    CapabilityMatrix,
    CapabilityMatrixRow,
    capability_matrix,
)
from agentbridge.compare import BackendComparison, compare_backends
from agentbridge.manifest import AgentManifest, load_agent_spec, load_manifest
from agentbridge.plugins import load_adapter_plugins, plugin_status
from agentbridge.registry import (
    adapter_sources,
    get_adapter,
    inspect_backend,
    inspect_backends,
    list_adapters,
    register_adapter,
    reset_adapters,
)
from agentbridge.runner import run_agent, stream_agent
from agentbridge.tool_registry import ToolRegistry, get_tool, list_tools, register_tool
from agentbridge.validation import ManifestValidation, validate_manifest
from agentbridge.versioning import dependency_versions
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult, ToolSpec

__all__ = [
    "AgentEvent",
    "AgentManifest",
    "AgentSpec",
    "BackendComparison",
    "BackendCapabilities",
    "CANONICAL_CAPABILITIES",
    "CapabilityFeature",
    "CapabilityMatrix",
    "CapabilityMatrixRow",
    "ManifestValidation",
    "RunInput",
    "RunResult",
    "ToolRegistry",
    "ToolSpec",
    "adapter_sources",
    "capability_matrix",
    "compare_backends",
    "dependency_versions",
    "event_to_agui",
    "get_adapter",
    "inspect_backend",
    "inspect_backends",
    "load_agent_spec",
    "load_manifest",
    "load_adapter_plugins",
    "get_tool",
    "list_tools",
    "plugin_status",
    "list_adapters",
    "register_tool",
    "register_adapter",
    "reset_adapters",
    "run_agent",
    "stream_agent",
    "validate_manifest",
]
