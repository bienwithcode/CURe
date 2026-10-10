from __future__ import annotations

import pytest


def test_config_disabled_resolution_and_non_table() -> None:
    from cure_system_one.config import resolve_system_one_config
    from cure_system_one.errors import SystemOneConfigError

    assert resolve_system_one_config(None).enabled is False
    assert resolve_system_one_config({}).enabled is False
    assert (
        resolve_system_one_config(
            {"enabled": False, "provider": "typesafe", "timeout_ms": -5}
        ).enabled
        is False
    )

    with pytest.raises(SystemOneConfigError):
        resolve_system_one_config("not-a-table")


@pytest.mark.parametrize(
    "section",
    [
        {"enabled": True, "provider": "typesafe"},
        {"enabled": True, "provider": "openai"},
        {"enabled": 1},
        {"enabled": "yes"},
        {"enabled": True, "grounding_confidence_min": 1.5},
        {"enabled": True, "grounding_confidence_min": -0.1},
        {"enabled": True, "grounding_confidence_min": "high"},
        {"enabled": True, "finding_acceptance_probability": -0.01},
        {"enabled": True, "finding_acceptance_probability": "many"},
        {"enabled": True, "timeout_ms": 0},
        {"enabled": True, "timeout_ms": -1},
        {"enabled": True, "timeout_ms": "2500"},
        {"enabled": True, "dedup_budget_ms": 0},
        {"enabled": True, "preflight_budget_ms": -5},
        {"enabled": True, "max_claim_chars": 0},
        {"enabled": True, "max_evidence_chars": -1},
        {"enabled": True, "max_context_chars": 0},
        {"enabled": True, "max_subject_chars": 0},
        {"enabled": True, "batch_chunk_size": 0},
        {"enabled": True, "temperature": -0.1},
        {"enabled": True, "temperature": "hot"},
        {"enabled": True, "model": 123},
        {"enabled": True, "api_key_env": 5},
    ],
)
def test_config_value_contract_rejects_invalid(section: dict) -> None:
    from cure_system_one.config import resolve_system_one_config
    from cure_system_one.errors import SystemOneConfigError

    with pytest.raises(SystemOneConfigError):
        resolve_system_one_config(section)

    unknown = resolve_system_one_config(
        {"enabled": True, "provider": "local", "not_a_system_one_key": "x"}
    )
    assert unknown.enabled is True
    assert unknown.provider == "local"


def test_config_inclusive_boundaries_and_frozen_snapshot() -> None:
    import dataclasses

    from cure_system_one.config import (
        DEFAULT_BATCH_CHUNK_SIZE,
        DEFAULT_MAX_CLAIM_CHARS,
        resolve_system_one_config,
    )
    from cure_system_one.errors import SystemOneConfigError

    resolved = resolve_system_one_config(
        {
            "enabled": True,
            "provider": "local",
            "model": "M1",
            "finding_acceptance_probability": 0.0,
            "grounding_confidence_min": 1.0,
        }
    )
    assert resolved.finding_acceptance_probability == 0.0
    assert resolved.grounding_confidence_min == 1.0
    assert resolved.provider == "local"
    assert resolved.model == "M1"
    assert resolved.max_claim_chars == DEFAULT_MAX_CLAIM_CHARS
    assert resolved.batch_chunk_size == DEFAULT_BATCH_CHUNK_SIZE

    assert dataclasses.is_dataclass(resolved)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.provider = "openrouter-decisions"  # type: ignore[misc]

    with pytest.raises(SystemOneConfigError):
        resolve_system_one_config({"enabled": True, "timeout_ms": 0})


def test_extract_system_one_section_is_pure_and_tolerant() -> None:
    from cure_system_one.config import extract_system_one_section

    section = {"enabled": True, "provider": "local"}
    # A parsed mapping yields its [system_one] table verbatim.
    assert extract_system_one_section({"system_one": section}) is section
    # Absent section, absent key, and non-mapping config all resolve to None.
    assert extract_system_one_section({}) is None
    assert extract_system_one_section({"other": 1}) is None
    assert extract_system_one_section(None) is None
    assert extract_system_one_section("not-a-table") is None


def test_default_grounding_threshold_matches_calibrated_decisions() -> None:
    # The shipped default provider is the calibrated decisions adapter, so the
    # grounding floor must sit below its truth probabilities (empirically
    # 0.62-0.77), not at the chat-model-era 0.80.
    from cure_system_one.config import (
        DEFAULT_GROUNDING_CONFIDENCE_MIN,
        DEFAULT_PROVIDER,
        resolve_system_one_config,
    )

    assert DEFAULT_PROVIDER == "openrouter-decisions"
    assert DEFAULT_GROUNDING_CONFIDENCE_MIN == 0.6
    resolved = resolve_system_one_config({"enabled": True})
    assert resolved.grounding_confidence_min == 0.6
