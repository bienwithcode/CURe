"""Two-layer citation verification and deterministic evidence-matrix serialization.

The stage verifies each representative finding's citations mechanically first —
parsing the ``Sources:`` ``path:line`` contract and confirming every path exists
in the sandbox and every line lies within the file — and only then semantically,
by asking the ``noul`` primitive whether the finding is supported by the cited
line. A finding is excluded when any of its citations fails either layer; the
surviving findings become the evidence matrix, serialized verbatim.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from cure_citations import (
    CITATION_LINE_RE,
    has_incomplete_sources,
    trailing_sources_suffix,
)
from cure_system_one.client import (
    PROVIDER_FAILURE_KINDS,
    passes_grounding_confidence,
    utc_now_iso,
)
from cure_system_one.config import SystemOneConfig
from cure_system_one.dedup import DedupCluster
from cure_system_one.errors import (
    SystemOneConfigError,
    SystemOneProviderAuthError,
    SystemOneSchemaViolationError,
    SystemOneTransientError,
)

MECHANICAL_LAYER = "mechanical"
SEMANTIC_LAYER = "semantic"

CITATION_DEAD = "citation_dead"
CITATION_UNSUPPORTED = "citation_unsupported"
CITATION_SUPPORTED = "citation_supported"

DEFAULT_MATRIX_FILENAME = "system-one-evidence-matrix.json"

_PREVIEW_LIMIT = 160


@dataclass(frozen=True)
class CitationEntry:
    """One verified citation with its capped evidence snippet and confidence."""

    path: str
    line: int
    snippet: str
    truncated: bool
    confidence: float

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "snippet": self.snippet,
            "truncated": self.truncated,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class MatrixRow:
    """One surviving finding, verbatim, with its verified citations."""

    cluster_id: str
    severity: str
    finding_text: str
    citations: tuple[CitationEntry, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "severity": self.severity,
            "finding_text": self.finding_text,
            "citations": [citation.to_record() for citation in self.citations],
        }


@dataclass(frozen=True)
class FindingExclusion:
    """One finding dropped by pre-flight, with its layer, reason, and preview."""

    cluster_id: str
    step_index: int
    card_position: int
    layer: str
    reason: str
    preview: str

    def to_meta(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "step_index": self.step_index,
            "card_position": self.card_position,
            "layer": self.layer,
            "reason": self.reason,
            "preview": self.preview,
        }


@dataclass(frozen=True)
class PreflightSummary:
    """Successful pre-flight summary, ready to persist under session meta.

    ``citations_checked`` counts the citations that reached the semantic layer
    (one ``noul`` call each); mechanically-dead citations are never submitted.
    """

    findings_in: int
    citations_checked: int
    mechanical_excluded: int
    semantic_excluded: int
    matrix_rows: int
    budget_ms_consumed: int
    provider: str

    def to_record(self) -> dict[str, Any]:
        return {
            "findings_in": self.findings_in,
            "citations_checked": self.citations_checked,
            "mechanical_excluded": self.mechanical_excluded,
            "semantic_excluded": self.semantic_excluded,
            "matrix_rows": self.matrix_rows,
            "budget_ms_consumed": self.budget_ms_consumed,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class PreflightDegraded:
    """One degraded record explaining why the whole stage was skipped.

    ``stage`` names the stage that degraded and ``recorded_at`` carries the
    ISO-8601 UTC instant it was created, matching the additive session-meta
    degraded-entry contract.
    """

    cause: str
    kind: str
    provider: str
    operation: str | None = None
    stage: str = "preflight"
    recorded_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of one pre-flight run.

    ``rows`` carries the serialized evidence matrix; ``exclusions`` names every
    dropped finding; ``degraded`` is set only when the whole stage was skipped
    (raw artifacts then feed synthesis), in which case ``summary`` is ``None``.
    """

    rows: tuple[MatrixRow, ...] = ()
    exclusions: tuple[FindingExclusion, ...] = ()
    summary: PreflightSummary | None = None
    degraded: PreflightDegraded | None = None
    artifact_path: str | None = None

    @property
    def skipped(self) -> bool:
        return self.degraded is not None

    def summary_record(self) -> dict[str, Any] | None:
        if self.summary is None:
            return None
        return self.summary.to_record()

    def exclusion_records(self) -> list[dict[str, Any]]:
        return [exclusion.to_meta() for exclusion in self.exclusions]

    def degraded_record(self) -> dict[str, Any] | None:
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
        return {
            "summary": self.summary_record(),
            "exclusions": self.exclusion_records(),
            "degraded": self.degraded_record(),
            "matrix_path": self.artifact_path,
        }


@dataclass(frozen=True)
class _Candidate:
    path: str
    line: int
    snippet: str
    truncated: bool


def run_preflight(
    clusters: Sequence[DedupCluster],
    *,
    client: Any,
    config: SystemOneConfig,
    repo_dir: str | Path,
    session_dir: str | Path | None = None,
    matrix_filename: str = DEFAULT_MATRIX_FILENAME,
    clock: Callable[[], float] = time.monotonic,
    budget_ms: int | None = None,
) -> PreflightResult:
    """Verify every cluster's citations and assemble the evidence matrix.

    The mechanical layer parses each representative's ``Sources:`` citations and
    confirms the sandbox file and line range; the semantic layer then asks
    ``noul`` for each mechanically-passing citation. A finding is excluded when
    any citation fails either layer.
    """
    effective_budget = config.preflight_budget_ms if budget_ms is None else budget_ms
    started = clock()
    root = Path(repo_dir)
    content_cache: dict[str, list[str] | None] = {}
    rows: list[MatrixRow] = []
    exclusions: list[FindingExclusion] = []
    mechanical_excluded = 0
    semantic_excluded = 0
    citations_checked = 0

    try:
        for cluster in clusters:
            card = cluster.representative
            candidates = _mechanical_citations(
                card.text, root, config.max_evidence_chars, cache=content_cache
            )
            if not candidates:
                mechanical_excluded += 1
                exclusions.append(
                    _exclusion(cluster, MECHANICAL_LAYER, CITATION_DEAD)
                )
                continue
            entries: list[CitationEntry] = []
            supported = True
            for candidate in candidates:
                if int((clock() - started) * 1000.0) >= effective_budget:
                    return _budget_skip(config)
                result = client.noul(card.text, candidate.snippet)
                citations_checked += 1
                confidence = float(result["probability"])
                if not passes_grounding_confidence(
                    confidence, config.grounding_confidence_min
                ):
                    supported = False
                    break
                entries.append(
                    CitationEntry(
                        path=candidate.path,
                        line=candidate.line,
                        snippet=candidate.snippet,
                        truncated=candidate.truncated,
                        confidence=confidence,
                    )
                )
            if not supported:
                semantic_excluded += 1
                exclusions.append(
                    _exclusion(cluster, SEMANTIC_LAYER, CITATION_UNSUPPORTED)
                )
                continue
            rows.append(
                MatrixRow(
                    cluster_id=cluster.cluster_id,
                    severity=card.severity,
                    finding_text=card.text,
                    citations=tuple(entries),
                )
            )
    except SystemOneConfigError as exc:
        return _skip(
            cause="config",
            kind="config",
            provider=exc.provider or config.provider,
            operation=exc.operation,
        )
    except (
        SystemOneTransientError,
        SystemOneProviderAuthError,
        SystemOneSchemaViolationError,
    ) as exc:
        return _skip(
            cause="provider_failure",
            kind=PROVIDER_FAILURE_KINDS[type(exc)],
            provider=exc.provider or config.provider,
            operation=exc.operation,
        )

    consumed = int((clock() - started) * 1000.0)
    if consumed >= effective_budget:
        return _budget_skip(config)
    summary = PreflightSummary(
        findings_in=len(clusters),
        citations_checked=citations_checked,
        mechanical_excluded=mechanical_excluded,
        semantic_excluded=semantic_excluded,
        matrix_rows=len(rows),
        budget_ms_consumed=consumed,
        provider=config.provider,
    )
    artifact_path: str | None = None
    if rows and session_dir is not None:
        artifact_path = _persist_matrix(rows, Path(session_dir), matrix_filename)
    return PreflightResult(
        rows=tuple(rows),
        exclusions=tuple(exclusions),
        summary=summary,
        artifact_path=artifact_path,
    )


def _skip(*, cause: str, kind: str, provider: str, operation: str | None) -> PreflightResult:
    return PreflightResult(
        degraded=PreflightDegraded(
            cause=cause,
            kind=kind,
            provider=provider,
            operation=operation,
        )
    )


def _budget_skip(config: SystemOneConfig) -> PreflightResult:
    return _skip(
        cause="budget_exhausted",
        kind="budget_exhausted",
        provider=config.provider,
        operation=None,
    )


def _exclusion(
    cluster: DedupCluster, layer: str, reason: str
) -> FindingExclusion:
    card = cluster.representative
    return FindingExclusion(
        cluster_id=cluster.cluster_id,
        step_index=card.step_index,
        card_position=card.card_position,
        layer=layer,
        reason=reason,
        preview=_preview_of(card.text),
    )


def _persist_matrix(
    rows: Sequence[MatrixRow], session_dir: Path, filename: str
) -> str:
    path = session_dir / filename
    _write_json_atomic(path, {"rows": [row.to_record() for row in rows]})
    return str(path)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as canonical JSON via an atomic replace.

    Keys are sorted and the encoding is fixed, so re-running the stage over
    byte-identical inputs produces a byte-identical artifact. This intentionally
    duplicates ``meta.write_json`` rather than importing it: ``cure_system_one``
    is a leaf package that may import only ``cure_errors`` and ``cure_citations``,
    so it must not reach into :mod:`meta`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".tmp-"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _preview_of(text: str) -> str:
    body = text[2:].strip() if text.startswith("- ") else text.strip()
    if len(body) > _PREVIEW_LIMIT:
        return body[: _PREVIEW_LIMIT - 3] + "..."
    return body


def _mechanical_citations(
    text: str,
    root: Path,
    cap: int,
    *,
    cache: dict[str, list[str] | None] | None = None,
) -> list[_Candidate]:
    """Return the mechanically-valid citations, or an empty list when dead.

    An empty result covers every mechanical failure: a missing ``Sources:``
    suffix, an incomplete or unparseable citation, a missing/empty/directory
    path, a line outside ``[1, file_line_count]``, a blank cited line, and a
    citation that escapes ``root``. ``cache`` optionally shares resolved file
    contents across citations within one run.
    """
    suffix = trailing_sources_suffix(text)
    if not suffix:
        return []
    if has_incomplete_sources(text):
        return []
    matches = list(CITATION_LINE_RE.finditer(suffix))
    if not matches:
        return []
    candidates: list[_Candidate] = []
    for match in matches:
        candidate = _resolve_citation(
            root, match.group(1), int(match.group(2)), cap, cache=cache
        )
        if candidate is None:
            return []
        candidates.append(candidate)
    return candidates


def _resolve_citation(
    root: Path,
    relative: str,
    line: int,
    cap: int,
    *,
    cache: dict[str, list[str] | None] | None = None,
) -> _Candidate | None:
    """Resolve one citation, rejecting anything outside ``root``.

    The root is resolved once and the citation target must stay inside it after
    symlink resolution, so ``../`` traversal and symlinks pointing out of the
    sandbox are dead citations rather than reads of external files. Resolved
    file contents are optionally cached so repeated citations do not re-read.
    """
    try:
        resolved_root = root.resolve()
        target = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if not target.is_relative_to(resolved_root):
        return None
    cache_key = str(target)
    if cache is not None and cache_key in cache:
        lines = cache[cache_key]
    else:
        try:
            if target.is_file():
                lines = target.read_text(encoding="utf-8").splitlines()
            else:
                lines = None
        except (OSError, UnicodeDecodeError, ValueError):
            lines = None
        if cache is not None:
            cache[cache_key] = lines
    if lines is None:
        return None
    if line < 1 or line > len(lines):
        return None
    snippet = lines[line - 1]
    if not snippet.strip():
        return None
    truncated = len(snippet) > cap
    if truncated:
        snippet = snippet[:cap]
    return _Candidate(
        path=relative, line=line, snippet=snippet, truncated=truncated
    )
