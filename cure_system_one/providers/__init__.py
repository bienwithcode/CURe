"""Provider adapter Protocol and registry for the System 1 core.

Adapters own transport only. They return ``(raw_dict, latency_ms)`` or raise
the internal :class:`ProviderTransportError`; the client owns schema
validation, retry, and the public error taxonomy. An adapter must never raise a
public :class:`cure_system_one.errors.SystemOneError`.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from cure_system_one.errors import SystemOneConfigError

_TRANSPORT_KINDS = ("transient", "auth")


class ProviderTransportError(Exception):
    """Internal transport failure raised by a provider adapter.

    ``kind`` is ``"transient"`` or ``"auth"``; the client maps it onto the
    public taxonomy. This class is deliberately not a ``SystemOneError`` so
    that adapters cannot accidentally bypass the client's classifier.
    """

    def __init__(self, kind: str, *, latency_ms: float = 0.0, message: str = "") -> None:
        if kind not in _TRANSPORT_KINDS:
            raise ValueError(
                f"Invalid provider transport error kind: {kind!r}. "
                f"Expected one of: {', '.join(_TRANSPORT_KINDS)}."
            )
        super().__init__(message or f"provider transport error: {kind}")
        self.kind = kind
        self.latency_ms = latency_ms


@runtime_checkable
class ProviderAdapter(Protocol):
    """Narrow transport seam: one provider call, no validation, no retry.

    ``supported_operations`` declares which primitives the adapter serves so
    the client can reject an unsupported operation with zero provider calls
    without knowing the provider by name.
    """

    supported_operations: frozenset[str]

    def complete(
        self,
        operation: str,
        request_payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Return ``(raw_dict, latency_ms)`` or raise ``ProviderTransportError``."""
        ...


def _local_factory(config: Any) -> ProviderAdapter:
    from cure_system_one.providers.local import LocalProvider

    return LocalProvider(config)


def _decisions_factory(config: Any) -> ProviderAdapter:
    from cure_system_one.providers.decisions import DecisionsAdapter

    return DecisionsAdapter(config)


_PROVIDER_FACTORIES: dict[str, Callable[[Any], ProviderAdapter]] = {
    "openrouter-decisions": _decisions_factory,
    "local": _local_factory,
}


def provider_names() -> tuple[str, ...]:
    """Return the registered provider names in registry order."""
    return tuple(_PROVIDER_FACTORIES)


def create_provider(provider_name: str, config: Any) -> ProviderAdapter:
    """Build the adapter registered under ``provider_name``.

    An unknown or non-string name raises ``SystemOneConfigError`` before any
    provider is constructed.
    """
    if not isinstance(provider_name, str) or provider_name not in _PROVIDER_FACTORIES:
        raise SystemOneConfigError(
            f"Unsupported provider in [system_one] config: {provider_name!r}. "
            f"Expected one of: {', '.join(provider_names())}."
        )
    return _PROVIDER_FACTORIES[provider_name](config)
