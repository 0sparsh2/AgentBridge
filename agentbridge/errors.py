"""AgentBridge exceptions."""


class AgentBridgeError(Exception):
    """Base error for AgentBridge."""


class AdapterNotFoundError(AgentBridgeError):
    """Raised when a backend adapter is not registered."""


class AdapterRegistrationError(AgentBridgeError):
    """Raised when an adapter cannot be registered."""


class MissingDependencyError(AgentBridgeError):
    """Raised when an optional backend dependency is not installed."""

    def __init__(self, backend: str, package: str, extra: str) -> None:
        message = (
            f"The '{backend}' backend requires the optional package '{package}'. "
            f"Install it with: pip install -e '.[{extra}]'"
        )
        super().__init__(message)
        self.backend = backend
        self.package = package
        self.extra = extra
