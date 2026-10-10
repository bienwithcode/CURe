"""Unit tests for the System 1 pre-synthesis pipeline assembly."""
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


def _client(
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


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _artifact(step_index: int, *bullets: str, path: str = ""):
    from cure_system_one.dedup import StepArtifact

    text = "### Findings\n" + "".join(f"{bullet}\n" for bullet in bullets)
    return StepArtifact(
        step_index=step_index,
        text=text,
        path=path or f"step-{step_index:02d}.md",
    )


def _probabilities(*values: float) -> dict[str, Any]:
    return {"noul": [{"value": {"probability": value}} for value in values]}


def test_pipeline_chain_and_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.pipeline import resolve_synth_inputs

    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "one\ntwo\nthree\n"})
    artifacts = [_artifact(0, "- [High] alpha. Sources: `src/a.py:1`")]

    # Dedup provider failure chains to a pre-flight skip: one degraded record per stage.
    failing = {"choice": [{"fault": "timeout"}, {"fault": "timeout"}]}
    client = _client(config, failing, tmp_path, monkeypatch)
    result = resolve_synth_inputs(
        artifacts,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "s1",
    )
    assert result.skipped is True
    assert result.matrix_path is None
    assert result.matrix_rows == 0
    assert result.exclusions == ()
    assert [record["cause"] for record in result.degraded] == [
        "provider_failure",
        "provider_failure",
    ]
    assert [record["kind"] for record in result.degraded] == ["transient", "transient"]

    # Pre-flight provider failure after a successful dedup skips only pre-flight.
    script = {"noul": [{"fault": "timeout"}, {"fault": "timeout"}]}
    client = _client(config, script, tmp_path, monkeypatch)
    result = resolve_synth_inputs(
        artifacts,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "s2",
    )
    assert result.skipped is True
    assert len(result.degraded) == 1
    assert result.degraded[0]["cause"] == "provider_failure"
    assert result.degraded[0]["kind"] == "transient"
    assert result.degraded[0]["operation"] == "noul"
    assert result.preflight_summary is None

    # Budget exhaustion skips the whole stage with budget_exhausted.
    client = _client(config, None, tmp_path, monkeypatch)
    result = resolve_synth_inputs(
        artifacts,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "s3",
        preflight_budget_ms=0,
    )
    assert result.skipped is True
    assert result.degraded[0]["cause"] == "budget_exhausted"
    assert result.degraded[0]["kind"] == "budget_exhausted"

    # Mechanical exclusions are finding-level outcomes and never fail the stage.
    client = _client(
        config, _probabilities(0.9), tmp_path, monkeypatch
    )
    result = resolve_synth_inputs(
        [
            _artifact(0, "- [High] valid. Sources: `src/a.py:1`"),
            _artifact(1, "- [High] dead. Sources: `src/a.py:99`"),
        ],
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "s4",
    )
    assert result.skipped is False
    assert result.matrix_rows == 1
    assert result.preflight_summary is not None
    assert result.preflight_summary["mechanical_excluded"] == 1


def test_pipeline_empty_matrix_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.pipeline import resolve_synth_inputs

    # An unscripted canned local provider answers noul with 0.5, below the
    # default 0.6 floor, so every mechanically-valid citation is excluded.
    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "one\ntwo\n"})
    client = _client(config, None, tmp_path, monkeypatch)
    session_dir = tmp_path / "empty"
    result = resolve_synth_inputs(
        [
            _artifact(0, "- [High] one. Sources: `src/a.py:1`"),
            _artifact(1, "- [Low] two. Sources: `src/a.py:2`"),
        ],
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=session_dir,
    )

    # The pipeline reports the empty signal and does not raise; the caller owns
    # the terminal action.
    assert result.skipped is False
    assert result.matrix_rows == 0
    assert result.matrix_path is None
    assert result.preflight_summary is not None
    assert result.preflight_summary["matrix_rows"] == 0
    assert result.preflight_summary["semantic_excluded"] == 2
    assert len(result.exclusions) == 2
    assert not list(session_dir.glob("*.json"))


def test_pipeline_zero_findings_skips_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one import pipeline
    from cure_system_one.dedup import StepArtifact
    from cure_system_one.pipeline import resolve_synth_inputs

    # A finding-free review must not run pre-flight: the stage runs only when
    # dedup produced clusters.
    calls = {"preflight": 0}

    def counting_preflight(*args: Any, **kwargs: Any) -> Any:
        calls["preflight"] += 1
        raise AssertionError("pre-flight must not run without clusters")

    monkeypatch.setattr(pipeline, "run_preflight", counting_preflight)

    config = _config()
    client = _client(config, None, tmp_path, monkeypatch)
    artifact = StepArtifact(
        step_index=0,
        text="### Findings\n\n### Suggested actions\n- none\n",
        path="step-00.md",
    )
    result = resolve_synth_inputs(
        [artifact],
        client=client,
        config=config,
        repo_dir=_repo(tmp_path, {}),
        session_dir=tmp_path / "zero",
    )

    assert result.skipped is False
    assert result.matrix_rows == 0
    assert result.findings_in == 0
    assert result.matrix_path is None
    assert result.preflight_summary is None
    assert result.degraded == ()
    assert calls["preflight"] == 0


def test_pipeline_meta_and_passthrough_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cure_system_one.dedup import StepArtifact
    from cure_system_one.pipeline import resolve_synth_inputs

    config = _config()
    repo = _repo(tmp_path, {"src/a.py": "one\ntwo\n"})
    broken = StepArtifact(
        step_index=2,
        text="### Findings\n- [High] ok\n\n```\n- stray\n",
        path="step-02.md",
    )
    artifacts = [
        _artifact(
            0,
            "- [High] alpha. Sources: `src/a.py:1`",
            "- [High] dead. Sources: `src/a.py:99`",
        ),
        broken,
    ]
    client = _client(config, _probabilities(0.9), tmp_path, monkeypatch)
    result = resolve_synth_inputs(
        artifacts,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "s",
    )

    assert result.skipped is False
    assert result.passthrough_paths == ("step-02.md",)
    summary = result.preflight_summary
    assert summary is not None
    assert set(summary) == {
        "findings_in",
        "citations_checked",
        "mechanical_excluded",
        "semantic_excluded",
        "matrix_rows",
        "budget_ms_consumed",
        "provider",
    }
    assert summary["findings_in"] == 2
    assert summary["citations_checked"] == 1
    assert summary["mechanical_excluded"] == 1
    assert summary["semantic_excluded"] == 0
    assert summary["matrix_rows"] == 1
    assert summary["provider"] == "local"
    assert 0 <= summary["budget_ms_consumed"] < config.preflight_budget_ms
    records = result.exclusion_records()
    assert records == [
        {
            "cluster_id": "c2",
            "step_index": 0,
            "card_position": 1,
            "layer": "mechanical",
            "reason": "citation_dead",
            "preview": "[High] dead. Sources: `src/a.py:99`",
        }
    ]
    meta = result.to_meta()
    assert meta["preflight"]["summary"]["matrix_rows"] == 1
    assert meta["preflight"]["exclusions"] == records
    assert meta["dedup"] is not None
    assert meta["degraded"] == []

    # A malformed script is a config error, which degrades both stages to skip.
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(bad_path))
    from cure_system_one.client import build_client

    result = resolve_synth_inputs(
        artifacts,
        client=build_client(config),
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "s2",
    )
    assert result.skipped is True
    assert [record["cause"] for record in result.degraded] == ["config", "config"]
    assert [record["kind"] for record in result.degraded] == ["config", "config"]
    assert result.preflight_summary is None


def test_preflight_replay_corpus_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Public entry point: this import only works once the assembly is exported.
    from cure_system_one import resolve_synth_inputs

    config = _config()
    lines = "\n".join(f"source line {index}" for index in range(1, 51)) + "\n"
    repo = _repo(tmp_path, {"src/a.py": lines})
    bullets = [
        "- [High] finding 01. Sources: `src/a.py:1`",
        "- [High] finding 02. Sources: `src/a.py:412`",
    ]
    bullets += [
        f"- [High] finding {index:02d}. Sources: `src/a.py:{index}`"
        for index in range(3, 13)
    ]
    artifacts = [_artifact(0, *bullets)]
    script = {
        "noul": [{"value": {"probability": 0.4}}]
        + [{"value": {"probability": 0.9}}] * 10
    }
    client = _client(config, script, tmp_path, monkeypatch)
    result = resolve_synth_inputs(
        artifacts,
        client=client,
        config=config,
        repo_dir=repo,
        session_dir=tmp_path / "replay",
    )

    summary = result.preflight_summary
    assert summary is not None
    assert summary["findings_in"] == 12
    assert summary["citations_checked"] == 11
    assert summary["mechanical_excluded"] == 1
    assert summary["semantic_excluded"] == 1
    assert summary["matrix_rows"] == 10
    assert result.matrix_rows == 10
    assert result.matrix_path is not None
    assert len(result.exclusions) == 2
    assert {entry.reason for entry in result.exclusions} == {
        "citation_dead",
        "citation_unsupported",
    }
    assert all(entry.preview for entry in result.exclusions)

    payload = json.loads(Path(result.matrix_path).read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 10
    assert [row["cluster_id"] for row in payload["rows"]] == [
        f"c{index}" for index in range(3, 13)
    ]
    assert [row["finding_text"] for row in payload["rows"]] == bullets[2:]
