"""Chain dedup, citation pre-flight, and the evidence matrix into synth inputs.

This module is the assembly the review wrapper calls: it runs dedup, and only
when dedup produced clusters does it run pre-flight. Any skip falls back to the
raw artifacts and records one degraded entry per skipped stage, so the caller
can degrade leniently without special-casing each failure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from cure_system_one.client import utc_now_iso
from cure_system_one.config import SystemOneConfig
from cure_system_one.dedup import StepArtifact, run_dedup
from cure_system_one.preflight import (
    DEFAULT_MATRIX_FILENAME,
    FindingExclusion,
    run_preflight,
)


@dataclass(frozen=True)
class SynthInputs:
    """Resolved synthesis inputs after the optional System 1 pre-flight.

    ``passthrough_paths`` names artifacts whose cards failed to parse and must
    re-enter synthesis byte-identically. ``matrix_path`` is the persisted
    evidence matrix (``None`` when the stage skipped or the matrix is empty);
    ``degraded`` carries one record per skipped stage.
    """

    passthrough_paths: tuple[str, ...] = ()
    matrix_path: str | None = None
    matrix_rows: int = 0
    findings_in: int = 0
    preflight_summary: dict[str, Any] | None = None
    exclusions: tuple[FindingExclusion, ...] = ()
    degraded: tuple[dict[str, Any], ...] = ()
    skipped: bool = False
    dedup: dict[str, Any] | None = None

    def exclusion_records(self) -> list[dict[str, Any]]:
        return [exclusion.to_meta() for exclusion in self.exclusions]

    def to_meta(self) -> dict[str, Any]:
        """Return the additive session-meta payload for both stages."""
        return {
            "dedup": self.dedup,
            "preflight": {
                "summary": self.preflight_summary,
                "exclusions": self.exclusion_records(),
            },
            "matrix_path": self.matrix_path,
            "degraded": [dict(record) for record in self.degraded],
        }


def resolve_synth_inputs(
    artifacts: Sequence[StepArtifact],
    *,
    client: Any,
    config: SystemOneConfig,
    repo_dir: str | Path,
    session_dir: str | Path | None = None,
    matrix_filename: str = DEFAULT_MATRIX_FILENAME,
    clock: Callable[[], float] = time.monotonic,
    dedup_budget_ms: int | None = None,
    preflight_budget_ms: int | None = None,
) -> SynthInputs:
    """Run dedup then pre-flight and assemble the synthesis inputs.

    A dedup skip chains to a pre-flight skip with one degraded record per
    skipped stage; a pre-flight skip records only its own degraded entry. In
    every skip case the raw artifacts remain the synthesis fallback and no
    matrix is produced.
    """
    dedup_result = run_dedup(
        artifacts,
        client=client,
        config=config,
        clock=clock,
        budget_ms=dedup_budget_ms,
    )
    if dedup_result.skipped:
        dedup_record = dedup_result.degraded_record() or {}
        return SynthInputs(
            matrix_path=None,
            matrix_rows=0,
            findings_in=0,
            preflight_summary=None,
            exclusions=(),
            degraded=(dedup_record, _mirror_degraded(dedup_record, config)),
            skipped=True,
            dedup=dedup_result.to_meta(),
        )

    passthrough_paths = tuple(
        artifact.path or f"step-{artifact.step_index}"
        for artifact in dedup_result.passthrough
    )
    findings_in = dedup_result.summary.findings_in if dedup_result.summary else 0

    # The stage runs only when dedup produced clusters. A finding-free review
    # proceeds to synthesis on the raw artifacts; it is not an all-excluded
    # matrix and must never trigger the empty-matrix terminal.
    if not dedup_result.clusters:
        return SynthInputs(
            passthrough_paths=passthrough_paths,
            matrix_path=None,
            matrix_rows=0,
            findings_in=findings_in,
            preflight_summary=None,
            exclusions=(),
            degraded=(),
            skipped=False,
            dedup=dedup_result.to_meta(),
        )

    preflight_result = run_preflight(
        dedup_result.clusters,
        client=client,
        config=config,
        repo_dir=repo_dir,
        session_dir=session_dir,
        matrix_filename=matrix_filename,
        clock=clock,
        budget_ms=preflight_budget_ms,
    )
    if preflight_result.skipped:
        preflight_record = preflight_result.degraded_record() or {}
        return SynthInputs(
            passthrough_paths=passthrough_paths,
            matrix_path=None,
            matrix_rows=0,
            findings_in=findings_in,
            preflight_summary=None,
            exclusions=(),
            degraded=(preflight_record,),
            skipped=True,
            dedup=dedup_result.to_meta(),
        )

    return SynthInputs(
        passthrough_paths=passthrough_paths,
        matrix_path=preflight_result.artifact_path,
        matrix_rows=len(preflight_result.rows),
        findings_in=findings_in,
        preflight_summary=preflight_result.summary_record(),
        exclusions=preflight_result.exclusions,
        degraded=(),
        skipped=False,
        dedup=dedup_result.to_meta(),
    )


def _mirror_degraded(
    dedup_record: dict[str, Any], config: SystemOneConfig
) -> dict[str, Any]:
    """Mirror the dedup skip cause onto the chained pre-flight skip."""
    return {
        "cause": dedup_record.get("cause"),
        "kind": dedup_record.get("kind"),
        "provider": dedup_record.get("provider") or config.provider,
        "operation": None,
        "stage": "preflight",
        "recorded_at": utc_now_iso(),
    }
