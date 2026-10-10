"""OpenRouter transport adapter for the System 1 core.

Stdlib ``urllib`` only. The adapter owns HTTP, JSON decode, and the internal
transport-error taxonomy; it never validates a response schema and never
raises a public ``SystemOneError``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from cure_system_one.providers import ProviderTransportError

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
# HTTP statuses that mean the OpenRouter credential was rejected. Shared with
# the runtime doctor probe so auth classification cannot drift between them.
OPENROUTER_AUTH_STATUSES = frozenset({401, 402, 403})


class OpenRouterAdapter:
    """POST chat-completion requests with strict JSON-schema structured output."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def complete(
        self,
        operation: str,
        request_payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], float]:
        body = self._build_body(operation, request_payload)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = self._timeout_seconds(deadline)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            latency = (time.monotonic() - started) * 1000.0
            raise ProviderTransportError(
                self._classify_status(exc.code), latency_ms=latency
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            latency = (time.monotonic() - started) * 1000.0
            raise ProviderTransportError("transient", latency_ms=latency) from exc
        latency = (time.monotonic() - started) * 1000.0
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
            raw = self._extract_payload(envelope)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ProviderTransportError(
                "transient", latency_ms=latency, message="malformed provider response"
            ) from exc
        return raw, latency

    def _build_body(
        self, operation: str, request_payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "model": self._config.model,
            "messages": list(request_payload.get("messages", [])),
            "temperature": self._config.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": operation,
                    "strict": True,
                    "schema": request_payload.get("schema", {}),
                },
            },
            "provider": {"require_parameters": True},
        }

    @staticmethod
    def _extract_payload(envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, dict):
            raise ValueError("provider response is not a JSON object")
        choices = envelope.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message") if isinstance(first, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                return json.loads(content)
            if isinstance(content, dict):
                return content
        return envelope

    @staticmethod
    def _classify_status(status_code: int) -> str:
        return "auth" if status_code in OPENROUTER_AUTH_STATUSES else "transient"

    def _api_key(self) -> str:
        return os.environ.get(self._config.api_key_env, "").strip()

    def _timeout_seconds(self, deadline: float | None) -> float:
        timeout = self._config.timeout_ms / 1000.0
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderTransportError(
                    "transient", latency_ms=0.0, message="provider deadline exceeded"
                )
            timeout = min(timeout, remaining)
        return timeout
