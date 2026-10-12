"""Generic OpenRouter decisions-modality transport adapter for the System 1 core.

Stdlib ``urllib`` only. The adapter owns HTTP, JSON decode, the mapping from a
provider-neutral ``{operation, input}`` request onto the decisions contract, and
the internal transport-error taxonomy. It never validates a primitive schema and
never raises a public ``SystemOneError``: an untranslatable answer is returned as
an ``{"error", "envelope"}`` payload (or ``{"error", "body"}`` for a non-JSON
body) that carries the served response and can never validate, so the client's
validator owns the schema verdict.

The adapter is deliberately model-generic: the ``model`` is read from
``[system_one].model`` so every decisions-modality model works by configuration
alone, with no per-model constraint table.
"""
from __future__ import annotations

import http.client
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
# Cap on the served body echoed into an error payload for debugging.
_RAW_BODY_CAP_CHARS = 4000


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
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            # ``http.client.HTTPException`` (e.g. ``IncompleteRead``) is not an
            # ``OSError``; a mid-response protocol failure is still transient.
            latency = (time.monotonic() - started) * 1000.0
            raise ProviderTransportError("transient", latency_ms=latency) from exc
        latency = (time.monotonic() - started) * 1000.0
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = raw_bytes.decode("utf-8", errors="replace")[:_RAW_BODY_CAP_CHARS]
            return {"error": "malformed provider response", "body": body}, latency
        fields = request_payload.get("input", {})
        if not isinstance(fields, dict):
            fields = {}
        return self._translate(operation, envelope, fields), latency

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
            # ``noul`` takes no criteria: the claim and evidence are the whole
            # question, and the endpoint rejects an empty criteria object.
            return {
                "type": "noul",
                "instructions": {
                    "claim": fields.get("claim", ""),
                    "evidence": fields.get("evidence", ""),
                },
            }
        if operation == "choice":
            return {
                "type": "choice",
                "instructions": {
                    "subject": fields.get("subject", ""),
                    "context": fields.get("context", ""),
                },
                "criteria": {
                    str(option): str(option) for option in fields.get("options", [])
                },
            }
        if operation == "score":
            rubric = fields.get("rubric", {})
            rubric = rubric if isinstance(rubric, dict) else {}
            # The endpoint requires ``score`` criteria to be an array; the two
            # rubric bounds are the scale endpoints the answer index maps onto.
            return {
                "type": "score",
                "instructions": {
                    "subject": fields.get("subject", ""),
                    "context": fields.get("context", ""),
                },
                "criteria": [str(rubric.get("min")), str(rubric.get("max"))],
            }
        return {"type": operation, "instructions": dict(fields)}

    # -- answer translation -------------------------------------------------

    @staticmethod
    def _translate(
        operation: str, envelope: Any, fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate a decisions envelope into the primitive raw shape.

        Any untranslatable response returns an ``{"error", "envelope"}`` payload
        rather than the bare envelope, so a body whose top-level keys happen to
        mimic a primitive answer can never validate as one, while the served
        envelope is still carried for the client's schema-violation record.
        """
        if not isinstance(envelope, dict):
            return _untranslatable("provider response is not a JSON object", envelope)
        answers = envelope.get("answers")
        if not isinstance(answers, dict):
            return _untranslatable("decisions response has no answers object", envelope)
        answer = answers.get(operation)
        if not isinstance(answer, dict) or answer.get("type") != operation:
            return _untranslatable(
                f"decisions response has no {operation!r} answer", envelope
            )
        if operation == "noul":
            # The decisions answer carries the probability under ``noul``; the
            # other keys are accepted for recorded/idealized envelopes.
            probability = answer.get("noul")
            if not _is_number(probability):
                probability = answer.get("probability")
            if not _is_number(probability):
                probability = _single_probability(answer.get("probabilities"))
            if not _is_number(probability):
                return _untranslatable("noul answer has no probability", envelope)
            return {"probability": probability}
        if operation == "choice":
            selected = answer.get("choice")
            confidence = answer.get("confidence")
            if not isinstance(selected, str) or not _is_number(confidence):
                return _untranslatable("choice answer is incomplete", envelope)
            return {"selected": selected, "confidence": confidence}
        if operation == "score":
            index = answer.get("score")
            confidence = answer.get("confidence")
            if not _is_number(index) or not _is_number(confidence):
                return _untranslatable("score answer is incomplete", envelope)
            value = _rubric_value(fields, index)
            if value is None:
                return _untranslatable("score answer has no usable rubric", envelope)
            return {"score": value, "confidence": confidence}
        return _untranslatable(f"unsupported operation {operation!r}", envelope)

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


def _untranslatable(reason: str, envelope: Any) -> dict[str, Any]:
    """Return an error payload that carries the served envelope verbatim."""
    return {"error": reason, "envelope": envelope}


def _is_number(value: Any) -> bool:
    # An oversized JSON int cannot convert to float; treat it as untranslatable.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _single_probability(probabilities: Any) -> Any:
    if not isinstance(probabilities, dict):
        return None
    values = [value for value in probabilities.values() if _is_number(value)]
    if len(values) == 1:
        return values[0]
    return None


def _rubric_value(fields: dict[str, Any], index: Any) -> float | None:
    """Map a decisions score index onto the rubric's inclusive ``min``..``max``.

    The endpoint returns ``score`` as the probability-weighted index into the
    two-entry criteria array, so ``0`` maps to ``min`` and ``1`` maps to
    ``max``. The index is clamped to ``[0, 1]`` so a stray value can never push
    the translated score outside the rubric the client validates against.
    """
    rubric = fields.get("rubric", {})
    if not isinstance(rubric, dict):
        return None
    minimum = rubric.get("min")
    maximum = rubric.get("max")
    if not _is_number(minimum) or not _is_number(maximum):
        return None
    ratio = float(index)
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0
    return float(minimum) + ratio * (float(maximum) - float(minimum))
