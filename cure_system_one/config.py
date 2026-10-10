"""``[system_one]`` configuration resolution for the System 1 core.

This module performs no I/O: it resolves an already-parsed TOML table into an
immutable :class:`SystemOneConfig` snapshot. A missing section, a missing
``enabled`` key, and ``enabled: false`` all resolve to a disabled snapshot, and
a disabled snapshot wins over otherwise-invalid values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cure_system_one.errors import SystemOneConfigError

DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_TIMEOUT_MS = 2500
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FINDING_ACCEPTANCE_PROBABILITY = 0.75
DEFAULT_GROUNDING_CONFIDENCE_MIN = 0.80
DEFAULT_DEDUP_BUDGET_MS = 120000
DEFAULT_PREFLIGHT_BUDGET_MS = 120000
DEFAULT_MAX_CLAIM_CHARS = 4000
DEFAULT_MAX_EVIDENCE_CHARS = 4000
DEFAULT_MAX_CONTEXT_CHARS = 4000
DEFAULT_MAX_SUBJECT_CHARS = 2000
DEFAULT_BATCH_CHUNK_SIZE = 100

SUPPORTED_PROVIDERS = ("openrouter", "local")


@dataclass(frozen=True)
class SystemOneConfig:
    """Immutable, per-stage System 1 configuration snapshot."""

    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    temperature: float = DEFAULT_TEMPERATURE
    finding_acceptance_probability: float = DEFAULT_FINDING_ACCEPTANCE_PROBABILITY
    grounding_confidence_min: float = DEFAULT_GROUNDING_CONFIDENCE_MIN
    dedup_budget_ms: int = DEFAULT_DEDUP_BUDGET_MS
    preflight_budget_ms: int = DEFAULT_PREFLIGHT_BUDGET_MS
    max_claim_chars: int = DEFAULT_MAX_CLAIM_CHARS
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_subject_chars: int = DEFAULT_MAX_SUBJECT_CHARS
    batch_chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE

    @classmethod
    def disabled(cls) -> "SystemOneConfig":
        return cls(enabled=False)


def resolve_system_one_config(raw_section: Any) -> SystemOneConfig:
    """Resolve ``[system_one]`` into a frozen snapshot.

    ``raw_section`` is the parsed TOML value for the ``system_one`` key (or
    ``None`` when the section is absent). Disabled resolution short-circuits
    before any value is validated, so ``enabled: false`` beside otherwise
    invalid values still resolves disabled. A non-table section is malformed.
    """
    if raw_section is None:
        return SystemOneConfig.disabled()
    if not isinstance(raw_section, dict):
        raise SystemOneConfigError(
            "Invalid [system_one] config. Expected a table of System 1 settings."
        )
    enabled = raw_section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise SystemOneConfigError(
            "Invalid enabled in [system_one] config. Expected a boolean."
        )
    if not enabled:
        return SystemOneConfig.disabled()

    provider = raw_section.get("provider", DEFAULT_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        raise SystemOneConfigError(
            f"Unsupported provider in [system_one] config: {provider!r}. "
            f"Expected one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    model = raw_section.get("model", DEFAULT_MODEL)
    if not isinstance(model, str):
        raise SystemOneConfigError(
            "Invalid model in [system_one] config. Expected a string."
        )

    api_key_env = raw_section.get("api_key_env", DEFAULT_API_KEY_ENV)
    if not isinstance(api_key_env, str):
        raise SystemOneConfigError(
            "Invalid api_key_env in [system_one] config. Expected a string."
        )

    return SystemOneConfig(
        enabled=True,
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        timeout_ms=_require_positive_int(raw_section, "timeout_ms", DEFAULT_TIMEOUT_MS),
        temperature=_require_temperature(raw_section),
        finding_acceptance_probability=_require_unit_interval(
            raw_section, "finding_acceptance_probability", DEFAULT_FINDING_ACCEPTANCE_PROBABILITY
        ),
        grounding_confidence_min=_require_unit_interval(
            raw_section, "grounding_confidence_min", DEFAULT_GROUNDING_CONFIDENCE_MIN
        ),
        dedup_budget_ms=_require_positive_int(
            raw_section, "dedup_budget_ms", DEFAULT_DEDUP_BUDGET_MS
        ),
        preflight_budget_ms=_require_positive_int(
            raw_section, "preflight_budget_ms", DEFAULT_PREFLIGHT_BUDGET_MS
        ),
        max_claim_chars=_require_positive_int(
            raw_section, "max_claim_chars", DEFAULT_MAX_CLAIM_CHARS
        ),
        max_evidence_chars=_require_positive_int(
            raw_section, "max_evidence_chars", DEFAULT_MAX_EVIDENCE_CHARS
        ),
        max_context_chars=_require_positive_int(
            raw_section, "max_context_chars", DEFAULT_MAX_CONTEXT_CHARS
        ),
        max_subject_chars=_require_positive_int(
            raw_section, "max_subject_chars", DEFAULT_MAX_SUBJECT_CHARS
        ),
        batch_chunk_size=_require_positive_int(
            raw_section, "batch_chunk_size", DEFAULT_BATCH_CHUNK_SIZE
        ),
    )


def _require_positive_int(raw_section: dict[str, Any], key: str, default: int) -> int:
    value = raw_section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemOneConfigError(
            f"Invalid {key} in [system_one] config. Expected a positive integer."
        )
    return value


def _require_unit_interval(raw_section: dict[str, Any], key: str, default: float) -> float:
    value = raw_section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemOneConfigError(
            f"Invalid {key} in [system_one] config. Expected a number in [0.0, 1.0]."
        )
    resolved = float(value)
    if resolved < 0.0 or resolved > 1.0:
        raise SystemOneConfigError(
            f"Invalid {key} in [system_one] config. Expected a number in [0.0, 1.0]."
        )
    return resolved


def _require_temperature(raw_section: dict[str, Any]) -> float:
    value = raw_section.get("temperature", DEFAULT_TEMPERATURE)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemOneConfigError(
            "Invalid temperature in [system_one] config. Expected a non-negative number."
        )
    resolved = float(value)
    if resolved < 0.0:
        raise SystemOneConfigError(
            "Invalid temperature in [system_one] config. Expected a non-negative number."
        )
    return resolved
