"""Framework-specific AgentBridge extension namespaces.

Extensions are where AgentBridge can adopt framework nuances without bloating the
portable AgentSpec core or pretending every framework has the same mental model.
"""

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension
from agentbridge.extensions.registry import ExtensionProfile, extension_profile, extension_profiles

__all__ = [
    "ExtensionProfile",
    "FrameworkExtension",
    "UnsupportedExtension",
    "extension_profile",
    "extension_profiles",
]
