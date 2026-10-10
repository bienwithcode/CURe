"""Unit tests for cross-step finding deduplication."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _config(**overrides: Any):
    from cure_system_one.config import resolve_system_one_config

    section = {"enabled": True, "provider": "local", "model": "M3"}
    section.update(overrides)
    return resolve_system_one_config(section)


def _build_client(config: Any, script: dict[str, Any] | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from cure_system_one.client import build_client

    if script is None:
        monkeypatch.delenv("CURE_SYSTEM_ONE_SCRIPT", raising=False)
    else:
        path = tmp_path / "script.json"
        path.write_text(json.dumps(script), encoding="utf-8")
        monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(path))
    return build_client(config)


def _artifact(step_index: int, *bullets: str, path: str = ""):
    from cure_system_one.dedup import StepArtifact

    text = "### Findings\n" + "".join(f"{bullet}\n" for bullet in bullets)
    return StepArtifact(step_index=step_index, text=text, path=path or f"step-{step_index:02d}.md")


def _choices(*decisions: tuple[str, float]) -> dict[str, Any]:
    return {
        "choice": [
            {"value": {"selected": selected, "confidence": confidence}}
            for selected, confidence in decisions
        ]
    }


class _RecordingClient:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[tuple[str, list[str]]] = []

    def choice(self, subject: str, options: list[str], *, context: str = "") -> dict[str, Any]:
        self.calls.append((subject, list(options)))
        return self._inner.choice(subject, options, context=context)


def test_merge_on_confident_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.dedup import run_dedup

    # Unscripted canned local provider selects the first option "new": zero merges.
    canned_config = _config()
    canned_client = _RecordingClient(
        _build_client(canned_config, None, tmp_path, monkeypatch)
    )
    canned = run_dedup(
        [_artifact(0, "- [High] duplicate", "- [High] duplicate")],
        client=canned_client,
        config=canned_config,
    )
    assert canned.skipped is False
    assert len(canned.clusters) == 2
    assert all(len(cluster.members) == 1 for cluster in canned.clusters)
    assert canned_client.calls[1][1] == ["new", "c1"]

    # Scripted decisions: merge only at or above the inclusive threshold.
    config = _config()
    script = _choices(("new", 0.5), ("c1", 0.9), ("c1", 0.75), ("c1", 0.6))
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    result = run_dedup(
        [
            _artifact(
                0,
                "- [High] duplicate",
                "- [High] duplicate",
                "- [High] duplicate",
                "- [High] duplicate",
            )
        ],
        client=client,
        config=config,
    )
    assert result.skipped is False
    assert [cluster.cluster_id for cluster in result.clusters] == ["c1", "c2"]
    assert [member.confidence for member in result.clusters[0].members] == [None, 0.9, 0.75]
    assert [member.confidence for member in result.clusters[1].members] == [None]
    assert [options for (_, options) in client.calls] == [
        ["new"],
        ["new", "c1"],
        ["new", "c1"],
        ["new", "c1"],
    ]


def test_representative_provenance_and_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.dedup import run_dedup

    config = _config()
    script = _choices(("new", 0.5), ("new", 0.5), ("c1", 0.9), ("c2", 0.9))
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    result = run_dedup(
        [
            _artifact(
                0,
                "- [High] alpha. Sources: `src/a.py:1`",
                "- [High] beta. Sources: `src/b.py:2`",
            ),
            _artifact(
                1,
                "- [High] alpha. Sources: `src/a.py:1`",
                "- [High] beta. Sources: `src/b.py:2`",
            ),
        ],
        client=client,
        config=config,
    )
    assert [cluster.cluster_id for cluster in result.clusters] == ["c1", "c2"]
    assert sum(len(cluster.members) for cluster in result.clusters) == 4
    assert (
        result.clusters[0].representative.text
        == "- [High] alpha. Sources: `src/a.py:1`"
    )
    assert (
        result.clusters[1].representative.text
        == "- [High] beta. Sources: `src/b.py:2`"
    )
    assert [
        (member.step_index, member.card_position)
        for member in result.clusters[0].members
    ] == [(0, 0), (1, 0)]
    assert [
        (member.step_index, member.card_position)
        for member in result.clusters[1].members
    ] == [(0, 1), (1, 1)]
    provenance = [
        (member.step_index, member.card_position)
        for cluster in result.clusters
        for member in cluster.members
    ]
    assert sorted(provenance) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_passthrough_and_degenerate_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.dedup import StepArtifact, run_dedup
    from cure_system_one.errors import SystemOneInputError

    config = _config()
    script = _choices(("new", 0.5))
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))

    good = _artifact(0, "- [High] only finding. Sources: `src/a.py:1`")
    empty = _artifact(1)
    broken_text = "### Findings\n- [High] ok\n\n```\n- stray unterminated\n"
    broken = StepArtifact(step_index=2, text=broken_text, path="step-02.md")

    result = run_dedup([good, empty, broken], client=client, config=config)
    assert len(result.clusters) == 1
    assert len(client.calls) == 1
    assert len(result.passthrough) == 1
    assert result.passthrough[0].text == broken_text
    assert result.passthrough[0].path == "step-02.md"

    missing = StepArtifact(step_index=0, text=None, path="step-00.md")
    with pytest.raises(SystemOneInputError):
        run_dedup([missing], client=client, config=config)


def _signature(result: Any) -> list[Any]:
    return [
        (
            cluster.cluster_id,
            cluster.representative.text,
            [
                (member.step_index, member.card_position, member.confidence)
                for member in cluster.members
            ],
        )
        for cluster in result.clusters
    ]


def test_canonical_order_and_idempotence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.dedup import run_dedup

    config = _config()
    step2 = _artifact(2, "- [High] shared defect. Sources: `src/a.py:1`")
    step4 = _artifact(4, "- [High] shared defect. Sources: `src/a.py:1`")

    def _run() -> Any:
        script = _choices(("new", 0.5), ("c1", 0.9))
        client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
        return run_dedup([step4, step2], client=client, config=config)

    first = _run()
    assert len(first.clusters) == 1
    assert first.clusters[0].representative.step_index == 2
    assert [
        (member.step_index, member.card_position)
        for member in first.clusters[0].members
    ] == [(2, 0), (4, 0)]

    second = _run()
    assert _signature(first) == _signature(second)


class _SteppingClock:
    def __init__(self, step_seconds: float) -> None:
        self._step = step_seconds
        self._value = 0.0
        self._first = True

    def __call__(self) -> float:
        if self._first:
            self._first = False
            return 0.0
        self._value += self._step
        return self._value


def test_skip_on_provider_failure_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.client import build_client
    from cure_system_one.dedup import StepArtifact, run_dedup

    corpus = [_artifact(0, "- [High] a", "- [High] b", "- [High] c")]
    config = _config()

    failing = {"choice": [{"fault": "timeout"}, {"fault": "timeout"}]}
    client = _RecordingClient(_build_client(config, failing, tmp_path, monkeypatch))
    result = run_dedup(corpus, client=client, config=config)
    assert result.skipped is True
    assert result.clusters == ()
    assert result.degraded.cause == "provider_failure"
    assert result.degraded.kind == "transient"
    assert result.degraded.provider == "local"
    assert result.degraded.operation == "choice"

    client = _RecordingClient(
        _build_client(config, {"choice": [{"fault": "http_401"}]}, tmp_path, monkeypatch)
    )
    result = run_dedup(corpus, client=client, config=config)
    assert result.skipped is True
    assert result.degraded.cause == "provider_failure"
    assert result.degraded.kind == "auth"

    client = _RecordingClient(
        _build_client(
            config, {"choice": [{"fault": "schema_violation"}]}, tmp_path, monkeypatch
        )
    )
    result = run_dedup(corpus, client=client, config=config)
    assert result.skipped is True
    assert result.degraded.kind == "schema_violation"

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(bad_path))
    client = _RecordingClient(build_client(config))
    result = run_dedup(corpus, client=client, config=config)
    assert result.skipped is True
    assert result.degraded.cause == "config"

    script = _choices(("new", 0.5), ("new", 0.5), ("new", 0.5), ("new", 0.5), ("new", 0.5))
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    result = run_dedup(
        corpus, client=client, config=config, clock=_SteppingClock(0.4), budget_ms=1000
    )
    assert result.skipped is True
    assert result.degraded.cause == "budget_exhausted"
    assert result.degraded.kind == "budget_exhausted"

    broken = StepArtifact(
        step_index=9,
        text="### Findings\n- [High] x\n\n```\n- stray unterminated\n",
        path="step-09.md",
    )
    client = _RecordingClient(
        _build_client(config, _choices(("new", 0.5)), tmp_path, monkeypatch)
    )
    result = run_dedup(
        [_artifact(0, "- [High] a"), broken], client=client, config=config
    )
    assert result.skipped is False
    assert len(result.passthrough) == 1


def test_meta_summary_and_degraded_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.dedup import run_dedup

    config = _config()
    script = _choices(("new", 0.5), ("new", 0.5), ("c1", 0.9), ("c2", 0.9))
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    result = run_dedup(
        [
            _artifact(0, "- [High] alpha", "- [High] beta"),
            _artifact(1, "- [High] alpha", "- [High] beta"),
        ],
        client=client,
        config=config,
    )
    assert result.degraded is None
    summary = result.summary
    assert summary is not None
    assert summary.findings_in == 4
    assert summary.clusters_out == 2
    assert summary.merged_count == 2
    assert summary.passthrough_artifacts == 0
    assert summary.provider == "local"
    assert 0 <= summary.budget_ms_consumed < config.dedup_budget_ms

    records = result.cluster_records()
    assert [record["id"] for record in records] == ["c1", "c2"]
    assert records[0]["provenance"] == [
        {"step_index": 0, "card_position": 0, "representative": True, "confidence": None},
        {"step_index": 1, "card_position": 0, "representative": False, "confidence": 0.9},
    ]
    assert records[0]["representative"] == "- [High] alpha"
    assert records[0]["confidence"] == [0.9]
    assert records[1]["provenance"][1]["confidence"] == 0.9
    assert result.to_meta()["summary"]["findings_in"] == 4
    assert result.to_meta()["degraded"] is None
    assert result.to_meta()["clusters"][0]["id"] == "c1"

    failing = {"choice": [{"fault": "timeout"}, {"fault": "timeout"}]}
    client = _RecordingClient(_build_client(config, failing, tmp_path, monkeypatch))
    skipped = run_dedup([_artifact(0, "- [High] a")], client=client, config=config)
    assert skipped.summary is None
    record = skipped.degraded_record()
    assert record is not None
    assert {key: record[key] for key in ("cause", "kind", "provider", "operation", "stage")} == {
        "cause": "provider_failure",
        "kind": "transient",
        "provider": "local",
        "operation": "choice",
        "stage": "dedup",
    }
    # ``recorded_at`` is captured when the stage degrades, in ISO-8601 UTC.
    assert record["recorded_at"] == skipped.degraded.recorded_at
    assert record["recorded_at"].endswith("+00:00")
    assert skipped.to_meta()["summary"] is None
    assert skipped.to_meta()["degraded"] == record


def test_replay_corpus_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Public entry point: this import only works once run_dedup is exported.
    from cure_system_one import run_dedup

    config = _config()
    flag_a = "- [High] shared auth bypass. Sources: `src/auth.py:10`"
    flag_b = "- [Medium] missing null check. Sources: `src/store.py:20`"
    flag_c = "- [Low] typo in message. Sources: `src/i18n.py:5`"
    flag_d = "- [Critical] unhandled exception. Sources: `src/run.py:99`"
    artifacts = [
        _artifact(0, flag_a, flag_b, flag_c, flag_d),
        _artifact(1, flag_a, flag_a, flag_b, flag_c),
        _artifact(2, flag_a, flag_b, flag_c, flag_d),
    ]
    script = _choices(
        ("new", 0.5),
        ("new", 0.5),
        ("new", 0.5),
        ("new", 0.5),
        ("c1", 0.9),
        ("c1", 0.8),
        ("c2", 0.9),
        ("c3", 0.9),
        ("c1", 0.9),
        ("c2", 0.9),
        ("c3", 0.9),
        ("c4", 0.9),
    )
    client = _RecordingClient(_build_client(config, script, tmp_path, monkeypatch))
    result = run_dedup(artifacts, client=client, config=config)

    assert result.skipped is False
    assert [cluster.cluster_id for cluster in result.clusters] == ["c1", "c2", "c3", "c4"]
    assert [cluster.representative.text for cluster in result.clusters] == [
        flag_a,
        flag_b,
        flag_c,
        flag_d,
    ]
    assert [
        (member.step_index, member.card_position)
        for member in result.clusters[0].members
    ] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    assert [member.confidence for member in result.clusters[0].members] == [
        None,
        0.9,
        0.8,
        0.9,
    ]
    assert result.summary is not None
    assert result.summary.findings_in == 12
    assert result.summary.clusters_out == 4
    assert result.summary.merged_count == 8
    assert result.summary.passthrough_artifacts == 0
    assert result.cluster_records()[0]["confidence"] == [0.9, 0.8, 0.9]

    # Identical sources with distinct claims stay separate.
    first = "- [High] first defect. Sources: `src/a.py:1`"
    second = "- [High] second defect. Sources: `src/a.py:1`"
    client = _RecordingClient(
        _build_client(config, _choices(("new", 0.5), ("new", 0.5)), tmp_path, monkeypatch)
    )
    distinct = run_dedup([_artifact(0, first, second)], client=client, config=config)
    assert len(distinct.clusters) == 2
    assert {cluster.representative.text for cluster in distinct.clusters} == {first, second}

    # A single-finding corpus is the identity transform.
    only = "- [High] only finding. Sources: `src/a.py:1`"
    client = _RecordingClient(
        _build_client(config, _choices(("new", 0.5)), tmp_path, monkeypatch)
    )
    single = run_dedup([_artifact(0, only)], client=client, config=config)
    assert len(single.clusters) == 1
    assert single.clusters[0].representative.text == only
    assert single.summary is not None
    assert single.summary.findings_in == 1

    # A total merge collapses to one verbatim representative with full provenance.
    client = _RecordingClient(
        _build_client(
            config,
            _choices(("new", 0.5), ("c1", 0.9), ("c1", 0.9), ("c1", 0.9), ("c1", 0.9)),
            tmp_path,
            monkeypatch,
        )
    )
    total = run_dedup([_artifact(0, "- [High] same", "- [High] same", "- [High] same", "- [High] same", "- [High] same")], client=client, config=config)
    assert len(total.clusters) == 1
    assert len(total.clusters[0].members) == 5
    assert total.clusters[0].representative.text == "- [High] same"
