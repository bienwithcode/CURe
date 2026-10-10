"""Generic OpenRouter decisions-modality transport adapter for the System 1 core.

Stdlib ``urllib`` only. The adapter owns HTTP, JSON decode, the mapping from a
provider-neutral ``{operation, input}`` request onto the decisions contract, and
the internal transport-error taxonomy. It never validates a primitive schema and
never raises a public ``SystemOneError``: an untranslatable answer is returned as
the raw envelope so the client's validator owns the schema verdict.

The adapter is deliberately model-generic: the ``model`` is read from
``[system_one].model`` so every decisions-modality model works by configuration
alone, with no per-model constraint table.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from cure_system_one.providers import ProviderTransportError

DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
# HTTP statuses that mean the OpenRouter credential was rejected. Shared with
# the runtime doctor probe so auth classification cannot drift between them.
DECISIONS_AUTH_STATUSES = frozenset({401, 402, 403})

_SUPPORTED_OPERATIONS = frozenset({"noul", "choice", "score"})


class DecisionsAdapter:
    """POST primitive questions to the OpenRouter decisions endpoint."""

    supported_operations = _SUPPORTED_OPERATIONS

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
            DECISIONS_ENDPOINT,
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
            if exc.code == 400:
                # A deterministic request rejection (invalid model or a per-model
                # criteria quirk): surface the raw error body so the client's
                # validator classifies it as a schema violation with no retry.
                return self._error_payload(exc), latency
            raise ProviderTransportError(
                self._classify_status(exc.code), latency_ms=latency
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            latency = (time.monotonic() - started) * 1000.0
            raise ProviderTransportError("transient", latency_ms=latency) from exc
        latency = (time.monotonic() - started) * 1000.0
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"error": "malformed provider response"}, latency
        return self._translate(operation, envelope), latency

    # -- request mapping ----------------------------------------------------

    def _build_body(
        self, operation: str, request_payload: dict[str, Any]
    ) -> dict[str, Any]:
        fields = request_payload.get("input", {})
        if not isinstance(fields, dict):
            fields = {}
        return {
            "model": self._config.model,
            "state": self._build_state(fields),
            "questions": {operation: self._build_question(operation, fields)},
        }

    @staticmethod
    def _build_state(fields: dict[str, Any]) -> Any:
        context = fields.get("context")
        return context if isinstance(context, str) else ""

    @staticmethod
    def _build_question(operation: str, fields: dict[str, Any]) -> dict[str, Any]:
        if operation == "noul":
            instructions: Any = {
                "claim": fields.get("claim", ""),
                "evidence": fields.get("evidence", ""),
            }
            criteria: Any = {}
        elif operation == "choice":
            instructions = {
                "subject": fields.get("subject", ""),
                "context": fields.get("context", ""),
            }
            criteria = {
                str(option): str(option) for option in fields.get("options", [])
            }
        elif operation == "score":
            instructions = {
                "subject": fields.get("subject", ""),
                "context": fields.get("context", ""),
            }
            rubric = fields.get("rubric", {})
            rubric = rubric if isinstance(rubric, dict) else {}
            criteria = {"min": rubric.get("min"), "max": rubric.get("max")}
        else:
            instructions = dict(fields)
            criteria = {}
        return {"type": operation, "instructions": instructions, "criteria": criteria}

    # -- answer translation -------------------------------------------------

    @staticmethod
    def _translate(operation: str, envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, dict):
            return {"error": "provider response is not a JSON object"}
        answers = envelope.get("answers")
        if not isinstance(answers, dict):
            return envelope
        answer = answers.get(operation)
        if not isinstance(answer, dict) or answer.get("type") != operation:
            return envelope
        if operation == "noul":
            probability = answer.get("probability")
            if not _is_number(probability):
                probability = _single_probability(answer.get("probabilities"))
            if not _is_number(probability):
                return envelope
            return {"probability": probability}
        if operation == "choice":
            selected = answer.get("choice")
            confidence = answer.get("confidence")
            if not isinstance(selected, str) or not _is_number(confidence):
                return envelope
            return {"selected": selected, "confidence": confidence}
        if operation == "score":
            score = answer.get("score")
            confidence = answer.get("confidence")
            if not _is_number(score) or not _is_number(confidence):
                return envelope
            return {"score": score, "confidence": confidence}
        return envelope

    # -- transport helpers --------------------------------------------------

    @staticmethod
    def _classify_status(status_code: int) -> str:
        return "auth" if status_code in DECISIONS_AUTH_STATUSES else "transient"

    @staticmethod
    def _error_payload(exc: urllib.error.HTTPError) -> dict[str, Any]:
        try:
            raw_bytes = exc.read()
            body = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return {"error": "decisions request rejected"}
        if isinstance(body, dict):
            return body
        return {"error": body}

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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _single_probability(probabilities: Any) -> Any:
    if not isinstance(probabilities, dict):
        return None
    values = [value for value in probabilities.values() if _is_number(value)]
    if len(values) == 1:
        return values[0]
    return None
