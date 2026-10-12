"""Live conformance probe for the OpenRouter decisions provider.

Requires a real ``OPENROUTER_API_KEY`` and network access; the whole module is
deselected by the suite's ``-k "not live"`` filter. It verifies the alpha
decisions endpoint contract end-to-end through :class:`SystemOneClient` and that
a deterministic request rejection surfaces as a schema violation, never a crash.
"""
from __future__ import annotations

import os

import pytest

from cure_system_one import build_client
from cure_system_one.config import SystemOneConfig
from cure_system_one.errors import SystemOneSchemaViolationError

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="live decisions probe requires OPENROUTER_API_KEY",
)

MODEL = "typesafe/jev-1.13"


def _live_config(model: str = MODEL) -> SystemOneConfig:
    return SystemOneConfig(
        enabled=True,
        provider="openrouter-decisions",
        model=model,
        timeout_ms=60000,
    )


def test_decisions_live_round_trip() -> None:
    client = build_client(_live_config())

    noul = client.noul(
        "The sky is blue on a clear day.",
        "Clear-sky observations are blue.",
        context="toy conformance probe",
    )
    assert 0.0 <= noul["probability"] <= 1.0

    choice = client.choice("Is 2 + 2 equal to 4?", ["yes", "no"])
    assert choice["selected"] in {"yes", "no"}
    assert 0.0 <= choice["confidence"] <= 1.0

    score = client.score("Rate the clarity of this sentence.", {"min": 0, "max": 10})
    assert 0.0 <= score["score"] <= 10.0
    assert 0.0 <= score["confidence"] <= 1.0


def test_decisions_live_rejection_is_schema_violation() -> None:
    # A chat-modality model is rejected by the decisions endpoint with HTTP 400;
    # the adapter returns the raw error body and the client classifies it as a
    # schema violation with no retry and no crash.
    client = build_client(_live_config(model="openai/gpt-4o-mini"))
    with pytest.raises(SystemOneSchemaViolationError) as error:
        client.noul("The sky is blue.", "Clear-sky observations are blue.")
    assert error.value.operation == "noul"
    assert error.value.attempts == 1
