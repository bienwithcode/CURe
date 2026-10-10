"""Cross-step finding deduplication over parsed finding cards.

The stage offers every finding to the ``choice`` primitive, ordered ``"new"``
first and existing cluster labels in formation order, and merges only when the
decision is confident enough. Clustering is deterministic over the artifacts'
content, not their arrival order.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from cure_system_one.cards import FindingCard, parse_findings
from cure_system_one.client import (
    PROVIDER_FAILURE_KINDS,
    passes_finding_acceptance,
    utc_now_iso,
)
from cure_system_one.config import SystemOneConfig
from cure_system_one.errors import (
    SystemOneConfigError,
    SystemOneInputError,
    SystemOneProviderAuthError,
    SystemOneSchemaViolationError,
    SystemOneTransientError,
)


@dataclass(frozen=True)
class StepArtifact:
    """One step artifact: its index, its text, and an optional audit path."""

    step_index: int
    text: str | None
    path: str = ""


@dataclass(frozen=True)
class ClusterMember:
    """One finding's provenance within a cluster.

    ``confidence`` is the merge decision confidence for members that joined an
    existing cluster and ``None`` for the canonical-first representative.
    """

    step_index: int
    card_position: int
    confidence: float | None = None


@dataclass(frozen=True)
class DedupCluster:
    """A group of duplicate findings keyed by first-appearance identity.

    ``representative`` is the canonical-first member's card, preserved
    byte-identically from its source artifact.
    """

    cluster_id: str
    representative: FindingCard
    members: tuple[ClusterMember, ...]


@dataclass(frozen=True)
class DedupDegraded:
    """One degraded record explaining why the whole stage was skipped.

    ``stage`` names the stage that degraded and ``recorded_at`` carries the
    ISO-8601 UTC instant it was created, matching the additive session-meta
    degraded-entry contract.
    """

    cause: str
    kind: str
    provider: str
    operation: str | None = None
    stage: str = "dedup"
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class DedupSummary:
    """Successful dedup summary, ready to persist under session meta."""

    findings_in: int
    clusters_out: int
    merged_count: int
    passthrough_artifacts: int
    budget_ms_consumed: int
    provider: str


@dataclass(frozen=True)
class DedupResult:
    """Outcome of one dedup run.

    ``passthrough`` carries artifacts whose cards failed to parse so callers
    can re-enter them into synthesis input byte-identically; ``degraded`` is set
    only when the whole stage was skipped (raw artifacts then feed synthesis),
    in which case ``summary`` is ``None``.
    """

    clusters: tuple[DedupCluster, ...] = ()
    passthrough: tuple[StepArtifact, ...] = ()
    summary: DedupSummary | None = None
    degraded: DedupDegraded | None = None

    @property
    def skipped(self) -> bool:
        return self.degraded is not None

    def summary_record(self) -> dict[str, Any] | None:
        """Return the meta summary record, or ``None`` when the stage skipped."""
        if self.summary is None:
            return None
        summary = self.summary
        return {
            "findings_in": summary.findings_in,
            "clusters_out": summary.clusters_out,
            "merged_count": summary.merged_count,
            "passthrough_artifacts": summary.passthrough_artifacts,
            "budget_ms_consumed": summary.budget_ms_consumed,
            "provider": summary.provider,
        }

    def cluster_records(self) -> list[dict[str, Any]]:
        """Return one meta entry per cluster, in first-appearance order."""
        records: list[dict[str, Any]] = []
        for cluster in self.clusters:
            records.append(
                {
                    "id": cluster.cluster_id,
                    "provenance": [
                        {
                            "step_index": member.step_index,
                            "card_position": member.card_position,
                            "representative": index == 0,
                            "confidence": member.confidence,
                        }
                        for index, member in enumerate(cluster.members)
                    ],
                    "representative": cluster.representative.text,
                    "confidence": [member.confidence for member in cluster.members[1:]],
                }
            )
        return records

    def degraded_record(self) -> dict[str, Any] | None:
        """Return the single degraded record, or ``None`` on success."""
        if self.degraded is None:
            return None
        degraded = self.degraded
        return {
            "cause": degraded.cause,
            "kind": degraded.kind,
            "provider": degraded.provider,
            "operation": degraded.operation,
            "stage": degraded.stage,
            "recorded_at": degraded.recorded_at,
        }

    def to_meta(self) -> dict[str, Any]:
        """Return the additive meta payload for the dedup stage."""
        return {
            "summary": self.summary_record(),
            "clusters": self.cluster_records(),
            "passthrough": [
                artifact.path or f"step-{artifact.step_index}"
                for artifact in self.passthrough
            ],
            "degraded": self.degraded_record(),
        }


def run_dedup(
    artifacts: Sequence[StepArtifact],
    *,
    client: Any,
    config: SystemOneConfig,
    clock: Callable[[], float] = time.monotonic,
    budget_ms: int | None = None,
) -> DedupResult:
    """Cluster the findings carried by ``artifacts`` via the ``choice`` primitive.

    Findings are offered sequentially in canonical order; a finding joins an
    existing cluster only when the decision selects it with confidence at or
    above ``config.finding_acceptance_probability``. Any other outcome opens a
    new cluster whose identifier follows first-appearance order.
    """
    cards: list[FindingCard] = []
    passthrough: list[StepArtifact] = []
    for artifact in artifacts:
        if artifact.text is None:
            raise SystemOneInputError(
                f"Missing step artifact text at step {artifact.step_index}; "
                "resolve missing artifacts before running dedup."
            )
        if not isinstance(artifact.text, str):
            raise SystemOneInputError(
                f"Invalid step artifact text at step {artifact.step_index}; "
                "expected a string."
            )
        parsed = parse_findings(artifact.text, step_index=artifact.step_index)
        if parsed.parse_failed:
            passthrough.append(artifact)
            continue
        cards.extend(parsed.cards)

    # Canonical order: step index ascending, then card position ascending.
    cards.sort(key=lambda card: (card.step_index, card.card_position))

    effective_budget = config.dedup_budget_ms if budget_ms is None else budget_ms
    started = clock()
    labels: list[str] = []
    working: list[list[ClusterMember]] = []
    representatives: list[FindingCard] = []
    try:
        for card in cards:
            if int((clock() - started) * 1000.0) >= effective_budget:
                return _budget_skip(passthrough, config)
            options = ["new"] + list(labels)
            decision = client.choice(card.text, options)
            selected = decision["selected"]
            confidence = float(decision["confidence"])
            target_index: int | None = None
            if selected != "new":
                for index, label in enumerate(labels):
                    if label == selected:
                        target_index = index
                        break
            if target_index is not None and passes_finding_acceptance(
                confidence, config.finding_acceptance_probability
            ):
                working[target_index].append(
                    ClusterMember(card.step_index, card.card_position, confidence)
                )
            else:
                labels.append(f"c{len(working) + 1}")
                representatives.append(card)
                working.append([ClusterMember(card.step_index, card.card_position, None)])
    except SystemOneConfigError as exc:
        return DedupResult(
            passthrough=tuple(passthrough),
            degraded=DedupDegraded(
                cause="config",
                kind="config",
                provider=exc.provider or config.provider,
                operation=exc.operation,
            ),
        )
    except (
        SystemOneTransientError,
        SystemOneProviderAuthError,
        SystemOneSchemaViolationError,
    ) as exc:
        return DedupResult(
            passthrough=tuple(passthrough),
            degraded=DedupDegraded(
                cause="provider_failure",
                kind=PROVIDER_FAILURE_KINDS[type(exc)],
                provider=exc.provider or config.provider,
                operation=exc.operation,
            ),
        )

    consumed = int((clock() - started) * 1000.0)
    if consumed >= effective_budget:
        return _budget_skip(passthrough, config)

    clusters = tuple(
        DedupCluster(
            cluster_id=labels[index],
            representative=representatives[index],
            members=tuple(members),
        )
        for index, members in enumerate(working)
    )
    summary = DedupSummary(
        findings_in=len(cards),
        clusters_out=len(clusters),
        merged_count=len(cards) - len(clusters),
        passthrough_artifacts=len(passthrough),
        budget_ms_consumed=consumed,
        provider=config.provider,
    )
    return DedupResult(
        clusters=clusters,
        passthrough=tuple(passthrough),
        summary=summary,
    )


def _budget_skip(passthrough: list[StepArtifact], config: SystemOneConfig) -> DedupResult:
    return DedupResult(
        passthrough=tuple(passthrough),
        degraded=DedupDegraded(
            cause="budget_exhausted",
            kind="budget_exhausted",
            provider=config.provider,
            operation="choice",
        ),
    )
