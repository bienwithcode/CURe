"""Unit tests for two-layer citation pre-flight and matrix serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _config(**overrides: Any):
    from cure_system_one.config import resolve_system_one_config

    section = {
        "enabled": True,
        "provider": "local",
        "model": "M4",
        "max_evidence_chars": 4000,
    }
    section.update(overrides)
    return resolve_system_one_config(section)


def _build_client(
    config: Any,
    script: dict[str, Any] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from cure_system_one.client import build_client

    if script is None:
        monkeypatch.delenv("CURE_SYSTEM_ONE_SCRIPT", raising=False)
    else:
        path = tmp_path / "script.json"
        path.write_text(json.dumps(script), encoding="utf-8")
        monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(path))
    return build_client(config)


class _RecordingClient:
    """Wraps a client and records every ``noul`` invocation."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.noul_calls: list[tuple[str, str]] = []

    def noul(self, claim: str, evidence: str, *, context: str = "") -> dict[str, Any]:
        self.noul_calls.append((claim, evidence))
        return self._inner.noul(claim, evidence, context=context)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _cluster(
    cluster_id: str,
    text: str,
    *,
    step_index: int = 0,
    card_position: int = 0,
    severity: str = "High",
):
    from cure_system_one.cards import FindingCard
    from cure_system_one.dedup import ClusterMember, DedupCluster

    card = FindingCard(
        step_index=step_index,
        card_position=card_position,
        text=text,
        severity=severity,
        citations=(),
    )
    return DedupCluster(
        cluster_id=cluster_id,
        representative=card,
        members=(ClusterMember(step_index, card_position, None),),
    )


def test_mechanical_layer_validity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cure_system_one.preflight import run_preflight

    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "line1\nline2\nline3\n"})
    (repo / "src" / "adir").mkdir()
    (repo / "src" / "empty.py").write_text("", encoding="utf-8")
    script = {"noul": [{"value": {"probability": 0.9}}] * 4}
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))

    clusters = [
        _cluster("c1", "- [High] valid last line. Sources: `src/a.py:3`"),
        _cluster("c2", "- [High] beyond eof. Sources: `src/a.py:4`"),
        _cluster("c3", "- [High] missing file. Sources: `src/nope.py:1`"),
        _cluster("c4", "- [High] empty file. Sources: `src/empty.py:1`"),
        _cluster("c5", "- [High] directory. Sources: `src/adir:1`"),
        _cluster("c6", "- [High] line zero. Sources: `src/a.py:0`"),
        _cluster("c7", "- [High] no citations"),
        _cluster("c8", "- [High] path only. Sources: `src/a.py`"),
    ]
    result = run_preflight(clusters, client=client, config=config, repo_dir=repo)

    assert result.skipped is False
    assert result.summary is not None
    assert result.summary.findings_in == 8
    assert result.summary.mechanical_excluded == 7
    assert result.summary.semantic_excluded == 0
    assert result.summary.matrix_rows == 1
    assert [row.cluster_id for row in result.rows] == ["c1"]
    assert len(result.exclusions) == 7
    assert {entry.layer for entry in result.exclusions} == {"mechanical"}
    assert {entry.reason for entry in result.exclusions} == {"citation_dead"}

    # Mechanical failures consume zero provider calls: only c1 reached noul.
    assert len(client.noul_calls) == 1
    assert client.noul_calls[0][1] == "line3"


def test_semantic_layer_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.preflight import run_preflight

    config = _config(grounding_confidence_min=0.8)
    repo = _repo(tmp_path, {"src/a.py": "alpha\nbeta\ngamma\n"})
    script = {
        "noul": [
            {"value": {"probability": 0.9}},
            {"value": {"probability": 0.8}},
            {"value": {"probability": 0.79}},
        ]
    }
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    clusters = [
        _cluster("c1", "- [High] supported. Sources: `src/a.py:1`"),
        _cluster("c2", "- [High] inclusive floor. Sources: `src/a.py:2`"),
        _cluster("c3", "- [High] below floor. Sources: `src/a.py:3`"),
        _cluster("c4", "- [High] dead no call. Sources: `src/a.py:99`"),
    ]
    result = run_preflight(clusters, client=client, config=config, repo_dir=repo)

    assert result.skipped is False
    assert [row.cluster_id for row in result.rows] == ["c1", "c2"]
    assert [row.citations[0].confidence for row in result.rows] == [0.9, 0.8]
    assert result.summary is not None
    assert result.summary.mechanical_excluded == 1
    assert result.summary.semantic_excluded == 1
    assert result.summary.matrix_rows == 2
    # Only mechanically-valid citations reach the semantic layer.
    assert len(client.noul_calls) == 3
    assert [(entry.cluster_id, entry.layer, entry.reason) for entry in result.exclusions] == [
        ("c3", "semantic", "citation_unsupported"),
        ("c4", "mechanical", "citation_dead"),
    ]


def test_exclusion_records_and_omission_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.preflight import run_preflight
    from cure_system_one.prompt_vars import render_preflight_omission_block

    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "one\ntwo\nthree\n"})
    script = {
        "noul": [
            {"value": {"probability": 0.4}},
            {"value": {"probability": 0.9}},
        ]
    }
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    clusters = [
        _cluster(
            "c1",
            "- [High] dead with three citations. "
            "Sources: `src/a.py:1`, `src/a.py:2`, `src/a.py:99`",
        ),
        _cluster(
            "c2",
            "- [Medium] weakly supported. Sources: `src/a.py:2`",
            step_index=1,
            card_position=3,
            severity="Medium",
        ),
        _cluster("c3", "- [Low] fine. Sources: `src/a.py:3`", severity="Low"),
    ]
    result = run_preflight(clusters, client=client, config=config, repo_dir=repo)

    assert result.summary is not None
    assert result.summary.findings_in == 3
    assert result.summary.mechanical_excluded == 1
    assert result.summary.semantic_excluded == 1
    assert result.summary.matrix_rows == 1
    # Finding-level exclusion: one entry per excluded finding, no partial sets.
    assert len(result.exclusions) == 2
    dead, unsupported = result.exclusions
    assert dead.cluster_id == "c1"
    assert dead.layer == "mechanical"
    assert dead.reason == "citation_dead"
    assert dead.preview.startswith("[High] dead with three citations.")
    assert unsupported.cluster_id == "c2"
    assert (unsupported.step_index, unsupported.card_position) == (1, 3)
    assert unsupported.layer == "semantic"
    assert unsupported.reason == "citation_unsupported"
    assert unsupported.preview == "[Medium] weakly supported. Sources: `src/a.py:2`"
    records = result.exclusion_records()
    assert [record["reason"] for record in records] == [
        "citation_dead",
        "citation_unsupported",
    ]
    assert records[0]["cluster_id"] == "c1"

    block = render_preflight_omission_block(result.exclusions)
    lines = block.splitlines()
    assert lines[0] == "### Omitted Findings (System 1 Pre-flight)"
    assert len(lines) == 3
    assert "mechanical" in lines[1] and "citation_dead" in lines[1]
    assert "dead with three citations." in lines[1]
    assert "semantic" in lines[2] and "citation_unsupported" in lines[2]
    assert "weakly supported." in lines[2]


def test_matrix_serialization_verbatim_and_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.preflight import run_preflight

    config = _config(max_evidence_chars=10)
    long_line = "0123456789ABCDEFGHIJ"
    repo = _repo(tmp_path, {"src/a.py": f"short\n{long_line}\n"})
    script = {
        "noul": [
            {"value": {"probability": 0.9}},
            {"value": {"probability": 0.9}},
        ]
    }
    clusters = [
        _cluster("c2", "- [High] second. Sources: `src/a.py:1`"),
        _cluster(
            "c1",
            "- [Critical] first. Sources: `src/a.py:2`",
            severity="Critical",
        ),
    ]
    session_dir = tmp_path / "session"
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    result = run_preflight(
        clusters,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=session_dir,
    )

    # Rows preserve cluster-id (first-appearance) order; text is verbatim.
    assert [row.cluster_id for row in result.rows] == ["c2", "c1"]
    assert [row.finding_text for row in result.rows] == [
        "- [High] second. Sources: `src/a.py:1`",
        "- [Critical] first. Sources: `src/a.py:2`",
    ]
    assert [row.severity for row in result.rows] == ["High", "Critical"]
    plain = result.rows[0].citations[0]
    assert (plain.path, plain.line, plain.snippet, plain.truncated) == (
        "src/a.py",
        1,
        "short",
        False,
    )
    assert plain.confidence == 0.9
    capped = result.rows[1].citations[0]
    assert (capped.snippet, capped.truncated) == ("0123456789", True)
    assert capped.confidence == 0.9

    assert result.artifact_path is not None
    artifact = Path(result.artifact_path)
    assert artifact.is_file()
    assert artifact == session_dir / "system-one-evidence-matrix.json"
    first_bytes = artifact.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["rows"][0]["finding_text"] == "- [High] second. Sources: `src/a.py:1`"
    assert payload["rows"][0]["citations"][0] == {
        "path": "src/a.py",
        "line": 1,
        "snippet": "short",
        "truncated": False,
        "confidence": 0.9,
    }
    assert payload["rows"][1]["citations"][0]["truncated"] is True

    # Re-running over byte-identical inputs with identical outcomes is byte-identical.
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    rerun = run_preflight(
        clusters,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=session_dir,
    )
    assert rerun.artifact_path == result.artifact_path
    assert Path(rerun.artifact_path).read_bytes() == first_bytes


def test_citation_path_traversal_is_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.preflight import run_preflight

    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "one\ntwo\n"})
    secret = "SECRET-OUTSIDE-CONTENT"
    (tmp_path / "outside.txt").write_text(f"{secret}\n", encoding="utf-8")
    script = {"noul": [{"value": {"probability": 0.9}}] * 4}
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    clusters = [
        _cluster("c1", "- [High] traversal. Sources: `../outside.txt:1`"),
        _cluster("c2", "- [High] nested. Sources: `src/../../outside.txt:1`"),
        _cluster("c3", "- [High] fine. Sources: `src/a.py:1`"),
    ]
    result = run_preflight(clusters, client=client, config=config, repo_dir=repo)

    # Citations escaping the sandbox are dead, never read, never promoted.
    assert result.summary is not None
    assert result.summary.mechanical_excluded == 2
    assert result.summary.matrix_rows == 1
    assert [row.cluster_id for row in result.rows] == ["c3"]
    assert {entry.reason for entry in result.exclusions} == {"citation_dead"}
    assert all(secret not in claim and secret not in evidence for claim, evidence in client.noul_calls)
    for row in result.rows:
        for citation in row.citations:
            assert secret not in citation.snippet


def test_resolved_citation_cache_reuses_file_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.preflight import run_preflight

    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "alpha\nbeta\ngamma\n"})
    script = {
        "noul": [
            {"value": {"probability": 0.9}},
            {"value": {"probability": 0.9}},
        ]
    }
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    reads = {"count": 0}
    real_read_text = Path.read_text

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "a.py":
            reads["count"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    clusters = [
        _cluster("c1", "- [High] first. Sources: `src/a.py:1`"),
        _cluster("c2", "- [High] second. Sources: `src/a.py:2`"),
    ]
    result = run_preflight(clusters, client=client, config=config, repo_dir=repo)

    assert result.summary is not None
    assert result.summary.matrix_rows == 2
    # Two citations to the same file read it once, thanks to the cache.
    assert reads["count"] == 1
