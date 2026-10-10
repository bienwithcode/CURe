"""Deterministic local System 1 provider.

The local provider answers every primitive with fixed canned defaults unless a
script (``CURE_SYSTEM_ONE_SCRIPT``) supplies scripted outcomes. It performs no
network I/O and never raises a public ``SystemOneError``: scripted transport
faults surface as the internal ``ProviderTransportError`` so the client owns the
public taxonomy, retry policy, and schema validation.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from cure_system_one.errors import SystemOneConfigError
from cure_system_one.providers import ProviderTransportError

_CANNED_PROBABILITY = 0.5
_CANNED_CONFIDENCE = 0.5

_FAULT_KINDS = {
    "timeout": "transient",
    "connection_reset": "transient",
    "http_429": "transient",
    "http_500": "transient",
    "http_401": "auth",
    "http_402": "auth",
    "http_403": "auth",
}

_VALID_OPERATIONS = ("noul", "choice", "score", "batch_filter")
_VALID_FAULTS = frozenset(_FAULT_KINDS) | {"schema_violation"}


class LocalProvider:
    """Deterministic provider driven by canned defaults and an optional script."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._loaded = False
        self._queues: dict[str, list[Any]] = {}

    def complete(
        self,
        operation: str,
        request_payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], float]:
        started = time.monotonic()
        with self._lock:
            self._ensure_loaded()
            entry = self._consume(operation)
        if entry is None:
            raw = self._canned(operation, request_payload)
        else:
            raw = self._materialize(operation, request_payload, entry)
        latency = (time.monotonic() - started) * 1000.0
        return raw, latency

    # -- script lifecycle ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        raw_path = os.environ.get("CURE_SYSTEM_ONE_SCRIPT", "").strip()
        if not raw_path:
            self._loaded = True
            return
        self._queues = self._read_script(raw_path)
        self._loaded = True

    def _read_script(self, raw_path: str) -> dict[str, list[Any]]:
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise SystemOneConfigError(
                f"Invalid CURE_SYSTEM_ONE_SCRIPT: cannot read {raw_path!r}."
            ) from exc
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemOneConfigError(
                f"Invalid CURE_SYSTEM_ONE_SCRIPT: {raw_path!r} is not valid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise SystemOneConfigError(
                "Invalid CURE_SYSTEM_ONE_SCRIPT: expected a JSON object of operations."
            )
        queues: dict[str, list[Any]] = {}
        for operation, entries in parsed.items():
            if operation not in _VALID_OPERATIONS:
                raise SystemOneConfigError(
                    f"Invalid CURE_SYSTEM_ONE_SCRIPT: unknown operation {operation!r}."
                )
            if not isinstance(entries, list):
                raise SystemOneConfigError(
                    f"Invalid CURE_SYSTEM_ONE_SCRIPT: {operation!r} must map to a list."
                )
            for entry in entries:
                self._validate_entry(operation, entry)
            queues[operation] = list(entries)
        return queues

    @staticmethod
    def _validate_entry(operation: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            raise SystemOneConfigError(
                f"Invalid CURE_SYSTEM_ONE_SCRIPT: {operation!r} entry must be an object."
            )
        if "fault" in entry:
            fault = entry["fault"]
            if fault not in _VALID_FAULTS:
                raise SystemOneConfigError(
                    f"Invalid CURE_SYSTEM_ONE_SCRIPT: unknown fault {fault!r} "
                    f"for {operation!r}."
                )
            return
        if "value" in entry:
            if not isinstance(entry["value"], dict):
                raise SystemOneConfigError(
                    f"Invalid CURE_SYSTEM_ONE_SCRIPT: {operation!r} value must be an object."
                )
            return
        raise SystemOneConfigError(
            f"Invalid CURE_SYSTEM_ONE_SCRIPT: {operation!r} entry must carry "
            "either 'fault' or 'value'."
        )

    def _consume(self, operation: str) -> Any:
        entries = self._queues.get(operation)
        if not entries:
            return None
        return entries.pop(0)

    # -- outcome materialization -------------------------------------------

    def _materialize(
        self, operation: str, request_payload: dict[str, Any], entry: Any
    ) -> dict[str, Any]:
        fault = entry.get("fault") if isinstance(entry, dict) else None
        if fault in _FAULT_KINDS:
            raise ProviderTransportError(_FAULT_KINDS[fault], latency_ms=0.0)
        if fault == "schema_violation":
            if "value" in entry:
                return entry["value"]
            return self._canned_invalid(operation, request_payload)
        return entry["value"]

    def _canned_invalid(
        self, operation: str, request_payload: dict[str, Any]
    ) -> dict[str, Any]:
        fields = request_payload.get("input", {})
        if operation == "noul":
            return {"probability": 1.7}
        if operation == "choice":
            return {"selected": "OPTION-X", "confidence": _CANNED_CONFIDENCE}
        if operation == "score":
            rubric = fields.get("rubric", {})
            maximum = float(rubric.get("max", 0.0))
            return {"score": maximum + 1.0, "confidence": _CANNED_CONFIDENCE}
        if operation == "batch_filter":
            return {"filtered": ["__foreign__"]}
        raise ProviderTransportError(
            "transient", latency_ms=0.0, message=f"unknown operation {operation!r}"
        )

    # -- canned defaults ----------------------------------------------------

    def _canned(
        self, operation: str, request_payload: dict[str, Any]
    ) -> dict[str, Any]:
        fields = request_payload.get("input", {})
        if operation == "noul":
            return {"probability": _CANNED_PROBABILITY}
        if operation == "choice":
            options = fields.get("options", [])
            return {"selected": options[0], "confidence": _CANNED_CONFIDENCE}
        if operation == "score":
            rubric = fields.get("rubric", {})
            midpoint = (float(rubric["min"]) + float(rubric["max"])) / 2.0
            return {"score": midpoint, "confidence": _CANNED_CONFIDENCE}
        if operation == "batch_filter":
            return {"filtered": list(fields.get("items", []))}
        raise ProviderTransportError(
            "transient", latency_ms=0.0, message=f"unknown operation {operation!r}"
        )
