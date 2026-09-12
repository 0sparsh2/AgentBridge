"""Framework-specific AgentBridge extension namespaces.

Extensions are where AgentBridge can adopt framework nuances without bloating the
portable AgentSpec core or pretending every framework has the same mental model.
"""

from agentbridge.extensions.base import FrameworkExtension, UnsupportedExtension
from agentbridge.extensions.google_adk import GoogleADKExtension
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge.extensions.openai_agents import OpenAIAgentsExtension
from agentbridge.extensions.registry import ExtensionProfile, extension_profile, extension_profiles
from agentbridge.extensions.strands import StrandsExtension

__all__ = [
    "ExtensionProfile",
    "FrameworkExtension",
    "GoogleADKExtension",
    "LangChainExtension",
    "OpenAIAgentsExtension",
    "StrandsExtension",
    "UnsupportedExtension",
    "extension_profile",
    "extension_profiles",
]
