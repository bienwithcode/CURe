"""Structured error taxonomy for the System 1 core.

Every error raised by :mod:`cure_system_one` descends from
:class:`SystemOneError`, which carries the observable context fields
(``provider``, ``operation``, ``latency_ms``, ``attempts``) that downstream
degraded-state records are built from.
"""
from __future__ import annotations

from typing import Any

from cure_errors import ReviewflowError


class SystemOneError(ReviewflowError):
    """Base class for every failure surfaced by the System 1 core."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        operation: str | None = None,
        latency_ms: float = 0.0,
        attempts: int = 0,
    ) -> None:
        super().__init__(message or self.__class__.__name__)
        self.provider = provider
        self.operation = operation
        self.latency_ms = latency_ms
        self.attempts = attempts


class SystemOneConfigError(SystemOneError):
    """Raised when ``[system_one]`` configuration is malformed or unsupported."""


class SystemOneInputError(SystemOneError):
    """Raised when caller input is structurally malformed before any provider call."""


class SystemOneTransientError(SystemOneError):
    """Raised when a transient transport failure survives the single retry."""


class SystemOneProviderAuthError(SystemOneError):
    """Raised when the provider rejects the request for auth/quota reasons."""


class SystemOneSchemaViolationError(SystemOneError):
    """Raised when an untrusted provider payload fails client-side validation."""

    def __init__(
        self,
        operation: str,
        raw_payload: Any,
        *,
        provider: str | None = None,
        latency_ms: float = 0.0,
        attempts: int = 1,
        message: str = "",
    ) -> None:
        super().__init__(
            message or f"System 1 schema violation for operation {operation!r}",
            provider=provider,
            operation=operation,
            latency_ms=latency_ms,
            attempts=attempts,
        )
        self.raw_payload = raw_payload
