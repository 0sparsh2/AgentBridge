"""Base types for framework-specific extensions."""

from __future__ import annotations

from typing import Any


class UnsupportedExtension(RuntimeError):
    """Raised when a framework extension is not available for a backend."""


class FrameworkExtension:
    """Base class for backend-specific extension helpers."""

    framework: str = "base"

    def __init__(self, *, raw: Any | None = None) -> None:
        self.raw = raw

    def require_raw(self) -> Any:
        """Return the raw native object or fail with a clear extension error."""

        if self.raw is None:
            raise UnsupportedExtension(
                f"{self.framework} extension requires a raw backend object, but none was provided."
            )
        return self.raw
