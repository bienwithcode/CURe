from __future__ import annotations

from cure_errors import ReviewflowError


def test_error_taxonomy_and_context_fields() -> None:
    from cure_system_one.errors import (
        SystemOneConfigError,
        SystemOneError,
        SystemOneInputError,
        SystemOneProviderAuthError,
        SystemOneSchemaViolationError,
        SystemOneTransientError,
    )

    error = SystemOneError(
        "boom",
        provider="openrouter",
        operation="noul",
        latency_ms=12.5,
        attempts=2,
    )
    assert isinstance(error, ReviewflowError)
    assert error.provider == "openrouter"
    assert error.operation == "noul"
    assert error.latency_ms == 12.5
    assert error.attempts == 2

    for subclass in (
        SystemOneConfigError,
        SystemOneInputError,
        SystemOneSchemaViolationError,
        SystemOneTransientError,
        SystemOneProviderAuthError,
    ):
        assert issubclass(subclass, SystemOneError)

    violation = SystemOneSchemaViolationError("noul", {"probability": 1.7})
    assert violation.operation == "noul"
    assert violation.raw_payload == {"probability": 1.7}
