"""System 1 client: schema validation, input contract, and provider invocation.

The client is the single owner of schema validation, the two-tier input
contract, and the public error taxonomy. Providers own transport only. A client
is built from one frozen :class:`~cure_system_one.config.SystemOneConfig`
snapshot and holds no mutable shared state between calls.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from cure_system_one.config import SystemOneConfig
from cure_system_one.errors import (
    SystemOneConfigError,
    SystemOneInputError,
    SystemOneProviderAuthError,
    SystemOneSchemaViolationError,
    SystemOneTransientError,
)
from cure_system_one.providers import (
    ProviderAdapter,
    ProviderTransportError,
    create_provider,
)

_OPERATION_INSTRUCTIONS = {
    "noul": "Return the probability that the claim is supported by the evidence.",
    "choice": "Select exactly one option and report your confidence.",
    "score": "Score the subject against the rubric and report your confidence.",
    "batch_filter": "Return the subset of items that match the criteria, in input order.",
}


def is_number(value: Any) -> bool:
    """True for int/float but not bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_unit_interval(value: Any) -> bool:
    """True when ``value`` is a number in the closed interval ``[0.0, 1.0]``."""
    return is_number(value) and 0.0 <= float(value) <= 1.0


def passes_finding_acceptance(probability: float, threshold: float) -> bool:
    """True when ``probability`` meets ``threshold`` inclusively."""
    return probability >= threshold


def passes_grounding_confidence(confidence: float, minimum: float) -> bool:
    """True when ``confidence`` meets ``minimum`` inclusively."""
    return confidence >= minimum


class SystemOneClient:
    """Frozen-snapshot client over a single :class:`ProviderAdapter`."""

    def __init__(self, config: SystemOneConfig, provider: ProviderAdapter) -> None:
        self._config = config
        self._provider = provider

    @classmethod
    def from_snapshot(
        cls,
        config: SystemOneConfig,
        *,
        provider: ProviderAdapter | None = None,
    ) -> "SystemOneClient":
        """Build a client over one frozen snapshot.

        When ``provider`` is omitted the adapter is resolved from the registry
        for the snapshot's provider name.
        """
        if provider is None:
            provider = create_provider(config.provider, config)
        return cls(config, provider)

    @property
    def config(self) -> SystemOneConfig:
        return self._config

    # -- primitives ---------------------------------------------------------

    def noul(self, claim: Any, evidence: Any, *, context: Any = "") -> dict[str, Any]:
        claim = _require_text(claim, "claim")
        evidence = _require_text(evidence, "evidence")
        context = self._context_value(context)
        claim, t1 = self._trim(claim, self._config.max_claim_chars)
        evidence, t2 = self._trim(evidence, self._config.max_evidence_chars)
        context, t3 = self._trim(context, self._config.max_context_chars)
        fields = {"claim": claim, "evidence": evidence, "context": context}
        payload = self._build_request("noul", fields, _noul_schema())
        validated = self._call("noul", payload, _validate_noul)
        probability = validated["probability"]
        return {
            "probability": probability,
            "passed": passes_finding_acceptance(
                probability, self._config.finding_acceptance_probability
            ),
            "truncated": t1 or t2 or t3,
        }

    def choice(
        self, subject: Any, options: Any, *, context: Any = ""
    ) -> dict[str, Any]:
        subject = _require_text(subject, "subject")
        options = _require_options(options)
        context = self._context_value(context)
        subject, t1 = self._trim(subject, self._config.max_subject_chars)
        context, t2 = self._trim(context, self._config.max_context_chars)
        fields = {"subject": subject, "options": list(options), "context": context}
        payload = self._build_request("choice", fields, _choice_schema(options))
        validated = self._call(
            "choice", payload, lambda raw: _validate_choice(raw, options)
        )
        return {
            "selected": validated["selected"],
            "confidence": validated["confidence"],
            "truncated": t1 or t2,
        }

    def score(
        self, subject: Any, rubric: Any, *, context: Any = ""
    ) -> dict[str, Any]:
        subject = _require_text(subject, "subject")
        minimum, maximum = _require_rubric(rubric)
        context = self._context_value(context)
        subject, t1 = self._trim(subject, self._config.max_subject_chars)
        context, t2 = self._trim(context, self._config.max_context_chars)
        fields = {
            "subject": subject,
            "rubric": {"min": minimum, "max": maximum},
            "context": context,
        }
        payload = self._build_request("score", fields, _score_schema(minimum, maximum))
        validated = self._call(
            "score", payload, lambda raw: _validate_score(raw, minimum, maximum)
        )
        return {
            "score": validated["score"],
            "confidence": validated["confidence"],
            "truncated": t1 or t2,
        }

    def batch_filter(
        self, items: Any, *, criteria: Any = "", context: Any = ""
    ) -> dict[str, Any]:
        items = _require_items(items)
        if not items:
            return {"filtered": [], "truncated": False}
        criteria = self._context_value(criteria)
        context = self._context_value(context)
        criteria, t1 = self._trim(criteria, self._config.max_claim_chars)
        context, t2 = self._trim(context, self._config.max_context_chars)
        chunk_size = self._config.batch_chunk_size
        chunks = [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]

        def run_chunk(chunk: list[str]) -> list[str]:
            fields = {"criteria": criteria, "context": context, "items": list(chunk)}
            payload = self._build_request("batch_filter", fields, _batch_schema())
            return self._call(
                "batch_filter", payload, lambda raw: _validate_batch(raw, chunk)
            )

        if len(chunks) == 1:
            chunk_results = [run_chunk(chunks[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = [executor.submit(run_chunk, chunk) for chunk in chunks]
                chunk_results = [future.result() for future in futures]
        filtered = [item for chunk_result in chunk_results for item in chunk_result]
        return {"filtered": filtered, "truncated": t1 or t2}

    # -- invocation ---------------------------------------------------------

    def _invoke(
        self, operation: str, request_payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float, int]:
        deadline = time.monotonic() + self._config.timeout_ms / 1000.0
        attempts = 0
        while True:
            attempts += 1
            try:
                raw, latency = self._provider.complete(
                    operation, request_payload, deadline=deadline
                )
            except ProviderTransportError as exc:
                if exc.kind == "auth":
                    raise SystemOneProviderAuthError(
                        f"System 1 provider auth failure for {operation!r}",
                        provider=self._config.provider,
                        operation=operation,
                        latency_ms=exc.latency_ms,
                        attempts=attempts,
                    ) from exc
                if attempts >= 2:
                    raise SystemOneTransientError(
                        f"System 1 transient provider failure for {operation!r}",
                        provider=self._config.provider,
                        operation=operation,
                        latency_ms=exc.latency_ms,
                        attempts=attempts,
                    ) from exc
                continue
            return raw, latency, attempts

    def _call(
        self,
        operation: str,
        request_payload: dict[str, Any],
        validate: Callable[[Any], Any],
    ) -> Any:
        raw, latency, attempts = self._invoke(operation, request_payload)
        try:
            return validate(raw)
        except (ValueError, TypeError, KeyError) as exc:
            raise SystemOneSchemaViolationError(
                operation,
                raw,
                provider=self._config.provider,
                latency_ms=latency,
                attempts=attempts,
            ) from exc

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _context_value(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise SystemOneInputError(
                "Invalid context for System 1 call. Expected a string."
            )
        return value

    @staticmethod
    def _trim(value: str, cap: int) -> tuple[str, bool]:
        if len(value) > cap:
            return value[:cap], True
        return value, False

    @staticmethod
    def _build_request(
        operation: str, fields: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": _OPERATION_INSTRUCTIONS.get(operation, "")},
                {
                    "role": "user",
                    "content": "\n".join(f"{key}: {value}" for key, value in fields.items()),
                },
            ],
            "schema": schema,
            "input": fields,
        }


# -- structural input contract ---------------------------------------------


def build_client(
    config: SystemOneConfig, *, provider: ProviderAdapter | None = None
) -> SystemOneClient:
    """Public factory: build a client from one resolved snapshot.

    A disabled snapshot is a programming error; runtime wiring must check
    ``config.enabled`` and degrade before calling this factory.
    """
    if not config.enabled:
        raise SystemOneConfigError("System 1 is disabled in [system_one] config.")
    return SystemOneClient.from_snapshot(config, provider=provider)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemOneInputError(
            f"Invalid {field} for System 1 call. Expected a non-empty string."
        )
    return value


def _require_options(options: Any) -> list[str]:
    if not isinstance(options, list) or not options:
        raise SystemOneInputError(
            "Invalid options for System 1 choice. Expected a non-empty list of strings."
        )
    for item in options:
        if not isinstance(item, str):
            raise SystemOneInputError(
                "Invalid options for System 1 choice. Expected a non-empty list of strings."
            )
    return options


def _require_items(items: Any) -> list[str]:
    if not isinstance(items, list):
        raise SystemOneInputError(
            "Invalid items for System 1 batch_filter. Expected a list of strings."
        )
    for item in items:
        if not isinstance(item, str):
            raise SystemOneInputError(
                "Invalid items for System 1 batch_filter. Expected a list of strings."
            )
    return items


def _require_rubric(rubric: Any) -> tuple[float, float]:
    if not isinstance(rubric, dict) or not rubric:
        raise SystemOneInputError(
            "Invalid rubric for System 1 score. Expected a non-empty table with min/max."
        )
    minimum = rubric.get("min")
    maximum = rubric.get("max")
    if not is_number(minimum) or not is_number(maximum):
        raise SystemOneInputError(
            "Invalid rubric for System 1 score. Expected numeric min and max."
        )
    resolved_min = float(minimum)
    resolved_max = float(maximum)
    if resolved_min > resolved_max:
        raise SystemOneInputError(
            "Invalid rubric for System 1 score. Expected min <= max."
        )
    return resolved_min, resolved_max


# -- schemas ---------------------------------------------------------------


def _noul_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"probability": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["probability"],
        "additionalProperties": False,
    }


def _choice_schema(options: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected": {"type": "string", "enum": list(options)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["selected", "confidence"],
        "additionalProperties": False,
    }


def _score_schema(minimum: float, maximum: float) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": minimum, "maximum": maximum},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["score", "confidence"],
        "additionalProperties": False,
    }


def _batch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"filtered": {"type": "array", "items": {"type": "string"}}},
        "required": ["filtered"],
        "additionalProperties": False,
    }


# -- validators ------------------------------------------------------------


def _validate_noul(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("noul response is not an object")
    probability = raw.get("probability")
    if not is_unit_interval(probability):
        raise ValueError("noul probability is not in [0.0, 1.0]")
    return {"probability": float(probability)}


def _validate_choice(raw: Any, options: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("choice response is not an object")
    selected = raw.get("selected")
    if not isinstance(selected, str) or selected not in options:
        raise ValueError("choice selected is not one of the submitted options")
    confidence = raw.get("confidence")
    if not is_unit_interval(confidence):
        raise ValueError("choice confidence is not in [0.0, 1.0]")
    return {"selected": selected, "confidence": float(confidence)}


def _validate_score(raw: Any, minimum: float, maximum: float) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("score response is not an object")
    score = raw.get("score")
    if not is_number(score) or float(score) < minimum or float(score) > maximum:
        raise ValueError("score is outside the rubric range")
    confidence = raw.get("confidence")
    if not is_unit_interval(confidence):
        raise ValueError("score confidence is not in [0.0, 1.0]")
    return {"score": float(score), "confidence": float(confidence)}


def _validate_batch(raw: Any, items: list[str]) -> list[str]:
    if not isinstance(raw, dict):
        raise ValueError("batch_filter response is not an object")
    filtered = raw.get("filtered")
    if not isinstance(filtered, list):
        raise ValueError("batch_filter filtered is not a list")
    seen: set[str] = set()
    last_index = -1
    for item in filtered:
        if not isinstance(item, str):
            raise ValueError("batch_filter filtered contains a non-string item")
        if item in seen:
            raise ValueError("batch_filter filtered contains a duplicate item")
        seen.add(item)
        try:
            index = items.index(item)
        except ValueError as exc:
            raise ValueError("batch_filter filtered contains a foreign item") from exc
        if index <= last_index:
            raise ValueError("batch_filter filtered is not in input order")
        last_index = index
    return list(filtered)
