"""Feature tests for the System 1 pre-synthesis wiring into the review flow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_config(tmp_path: Path, section: str = "") -> Path:
    path = tmp_path / "cure.toml"
    path.write_text(section, encoding="utf-8")
    return path


def _step_outputs(tmp_path: Path, count: int = 2) -> list[str]:
    outputs: list[str] = []
    for index in range(count):
        path = tmp_path / f"step-{index:02d}.md"
        path.write_text(
            f"### Findings\n- [High] finding {index}. Sources: `src/a.py:1`\n",
            encoding="utf-8",
        )
        outputs.append(str(path))
    return outputs


def test_wrapper_delegates_and_passthrough_reentry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import cure
    from cure_system_one.pipeline import SynthInputs
    from cure_system_one.prompt_vars import evidence_matrix_prompt_vars

    outputs = _step_outputs(tmp_path, count=2)

    delegated: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> tuple[list[str], str]:
        delegated["prepare"] = kwargs
        return list(outputs), "skipped"

    monkeypatch.setattr(cure, "_prepare_synth_inputs", fake_prepare)

    matrix_path = tmp_path / "system-one-evidence-matrix.json"
    matrix_path.write_text('{"rows": []}\n', encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_resolve(artifacts: Any, **kwargs: Any) -> SynthInputs:
        captured["artifacts"] = list(artifacts)
        captured["kwargs"] = kwargs
        return SynthInputs(
            passthrough_paths=("step-99.md",),
            matrix_path=str(matrix_path),
            matrix_rows=3,
            preflight_summary={"matrix_rows": 3},
        )

    monkeypatch.setattr(cure, "resolve_synth_inputs", fake_resolve)

    config_path = _write_config(
        tmp_path, '[system_one]\nenabled = true\nprovider = "local"\n'
    )
    meta: dict[str, Any] = {}
    synth, skipped, so_ctx = cure._prepare_synth_inputs_with_system_one(
        meta=meta,
        step_entries=[],
        session_id="session-1",
        session_dir=tmp_path,
        work_dir=tmp_path,
        review_md_path=tmp_path / "review.md",
        repo_dir=tmp_path,
        config_path=config_path,
    )

    # Existing guard is called unchanged, then the pipeline is delegated to.
    assert delegated["prepare"]["session_id"] == "session-1"
    assert synth == outputs
    assert skipped == "skipped"
    assert so_ctx is not None and so_ctx.matrix_rows == 3
    artifacts = captured["artifacts"]
    assert [(item.step_index, item.path) for item in artifacts] == [
        (0, outputs[0]),
        (1, outputs[1]),
    ]
    assert artifacts[0].text == Path(outputs[0]).read_text(encoding="utf-8")
    assert captured["kwargs"]["repo_dir"] == tmp_path

    raw = "\n".join(f"- `{path}`" for path in outputs)
    on = evidence_matrix_prompt_vars(so_ctx, raw_step_paths_text=raw)
    paths = on["STEP_OUTPUT_PATHS"].splitlines()
    assert paths[0] == f"- `{matrix_path}`"
    assert paths[1] == "- `step-99.md`"
    assert f"- `{outputs[0]}`" not in paths

    off = evidence_matrix_prompt_vars(None, raw_step_paths_text=raw)
    assert off["STEP_OUTPUT_PATHS"] == raw


def test_matrix_variant_selected_strictly(tmp_path: Path) -> None:
    import cure
    from cure_flows import render_prompt
    from cure_system_one.pipeline import SynthInputs

    template = cure.load_builtin_prompt_text("mrereview_gh_local_big_synth.md")
    matrix_path = tmp_path / "system-one-evidence-matrix.json"
    matrix_text = "MATRIX-UNIQUE-TOKEN-12345"
    matrix_path.write_text(matrix_text, encoding="utf-8")
    outputs = _step_outputs(tmp_path, count=1)
    raw = "\n".join(f"- `{path}`" for path in outputs)

    def render(so_ctx: Any) -> str:
        base = {"STEP_OUTPUT_PATHS": raw, "GROUNDING_SKIPPED_STEPS": "- None."}
        extra = cure._synth_extra_vars_with_system_one(
            base, so_ctx=so_ctx, raw_step_paths_text=raw
        )
        return render_prompt(
            template,
            base_ref_for_review="main",
            pr_url="https://github.com/o/r/pull/1",
            pr_number=1,
            gh_host="github.com",
            gh_owner="o",
            gh_repo_name="r",
            gh_repo="o/r",
            agent_desc="",
            extra_vars=extra,
        )

    # Flag on and both stages succeeded: the matrix variant carries the matrix.
    success = SynthInputs(
        matrix_path=str(matrix_path), matrix_rows=2, passthrough_paths=()
    )
    prompt = render(success)
    assert matrix_text in prompt
    assert f"- `{matrix_path}`" in prompt

    # Any stage skip selects the original raw path.
    skipped = SynthInputs(
        skipped=True, degraded=({"cause": "provider_failure"},)
    )
    prompt = render(skipped)
    assert matrix_text not in prompt
    assert raw in prompt
    assert f"- `{matrix_path}`" not in prompt

    # Flag absent uses the original raw path and references no matrix artifact.
    prompt = render(None)
    assert matrix_text not in prompt
    assert raw in prompt
    assert f"- `{matrix_path}`" not in prompt


@pytest.mark.parametrize(
    "template_name",
    ["mrereview_gh_local_big_synth.md", "mrereview_gh_local_big_resume_synth.md"],
)
def test_flag_off_synth_prompt_matches_original(template_name: str) -> None:
    import cure
    from cure_flows import render_prompt

    raw = "- `step-00.md`"
    template = cure.load_builtin_prompt_text(template_name)
    base = {"STEP_OUTPUT_PATHS": raw, "GROUNDING_SKIPPED_STEPS": "- None."}
    extra = cure._synth_extra_vars_with_system_one(
        base, so_ctx=None, raw_step_paths_text=raw
    )
    prompt = render_prompt(
        template,
        base_ref_for_review="main",
        pr_url="https://github.com/o/r/pull/1",
        pr_number=1,
        gh_host="github.com",
        gh_owner="o",
        gh_repo_name="r",
        gh_repo="o/r",
        agent_desc="",
        extra_vars=extra,
    )

    # Flag off renders the pre-feature layout: no System 1 text, and the raw
    # read list is followed directly by the grounding-skipped bullet.
    assert "System 1" not in prompt
    assert "Evidence matrix" not in prompt
    assert f"{raw}\n- Grounding-skipped step outputs excluded from synth:\n" in prompt


def test_empty_matrix_terminal(tmp_path: Path, monkeypatch: Any) -> None:
    import cure
    from cure_system_one.pipeline import SynthInputs

    outputs = _step_outputs(tmp_path, count=2)
    monkeypatch.setattr(
        cure, "_prepare_synth_inputs", lambda **kwargs: (list(outputs), "skipped")
    )
    monkeypatch.setattr(
        cure,
        "resolve_synth_inputs",
        lambda artifacts, **kwargs: SynthInputs(
            skipped=False, matrix_rows=0, matrix_path=None, findings_in=2
        ),
    )

    playbook: dict[str, Any] = {"calls": 0}

    def fake_playbook(**kwargs: Any) -> None:
        playbook["calls"] += 1
        playbook["kwargs"] = kwargs

    monkeypatch.setattr(cure, "_emit_multipass_grounding_failure_playbook", fake_playbook)

    config_path = _write_config(
        tmp_path, '[system_one]\nenabled = true\nprovider = "local"\n'
    )
    meta: dict[str, Any] = {}
    with pytest.raises(cure.ReviewflowError):
        cure._prepare_synth_inputs_with_system_one(
            meta=meta,
            step_entries=[],
            session_id="session-empty",
            session_dir=tmp_path,
            work_dir=tmp_path,
            review_md_path=tmp_path / "review.md",
            repo_dir=tmp_path,
            config_path=config_path,
        )

    assert meta["status"] == "error"
    assert meta["multipass"]["status"] == "step_failed"
    assert playbook["calls"] == 1
    assert playbook["kwargs"]["resume_from"] == "steps"


def test_passthrough_suppresses_empty_matrix_terminal(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import cure
    from cure_system_one.pipeline import SynthInputs

    outputs = _step_outputs(tmp_path, count=2)
    monkeypatch.setattr(
        cure, "_prepare_synth_inputs", lambda **kwargs: (list(outputs), "skipped")
    )
    so_ctx = SynthInputs(
        skipped=False,
        matrix_rows=0,
        matrix_path=None,
        findings_in=1,
        passthrough_paths=(outputs[1],),
    )
    monkeypatch.setattr(cure, "resolve_synth_inputs", lambda artifacts, **kwargs: so_ctx)
    playbook: dict[str, Any] = {"calls": 0}
    monkeypatch.setattr(
        cure,
        "_emit_multipass_grounding_failure_playbook",
        lambda **kwargs: playbook.__setitem__("calls", playbook["calls"] + 1),
    )
    config_path = _write_config(
        tmp_path, '[system_one]\nenabled = true\nprovider = "local"\n'
    )
    meta: dict[str, Any] = {}

    # A passthrough artifact is still a usable synthesis input, so an
    # all-excluded matrix must not abort the review.
    _, _, result = cure._prepare_synth_inputs_with_system_one(
        meta=meta,
        step_entries=[],
        session_id="session-passthrough",
        session_dir=tmp_path,
        work_dir=tmp_path,
        review_md_path=tmp_path / "review.md",
        repo_dir=tmp_path,
        config_path=config_path,
    )
    assert result is so_ctx
    assert meta.get("status") != "error"
    assert playbook["calls"] == 0


def test_undecodable_step_artifact_raises_reviewflow_error(tmp_path: Path) -> None:
    import cure

    artifact = tmp_path / "step-00.md"
    artifact.write_bytes(b"### Findings\n- \xff\xfe not utf-8\n")
    config_path = _write_config(
        tmp_path, '[system_one]\nenabled = true\nprovider = "local"\n'
    )
    with pytest.raises(cure.ReviewflowError, match="step-00.md"):
        cure._resolve_system_one_synth_inputs(
            synth_step_outputs=[str(artifact)],
            repo_dir=tmp_path,
            session_dir=tmp_path,
            config_path=config_path,
        )


def test_zero_findings_skips_terminal_and_proceeds(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import cure
    from cure_system_one.pipeline import SynthInputs

    # A finding-free review reports zero findings in (not an all-excluded
    # matrix); the wrapper must hand the raw artifacts to synthesis normally.
    outputs = _step_outputs(tmp_path, count=2)
    monkeypatch.setattr(
        cure, "_prepare_synth_inputs", lambda **kwargs: (list(outputs), "skipped")
    )
    monkeypatch.setattr(
        cure,
        "resolve_synth_inputs",
        lambda artifacts, **kwargs: SynthInputs(
            skipped=False, matrix_rows=0, matrix_path=None, findings_in=0
        ),
    )

    playbook: dict[str, Any] = {"calls": 0}

    def fake_playbook(**kwargs: Any) -> None:
        playbook["calls"] += 1

    monkeypatch.setattr(cure, "_emit_multipass_grounding_failure_playbook", fake_playbook)

    config_path = _write_config(
        tmp_path, '[system_one]\nenabled = true\nprovider = "local"\n'
    )
    meta: dict[str, Any] = {}
    synth, skipped, so_ctx = cure._prepare_synth_inputs_with_system_one(
        meta=meta,
        step_entries=[],
        session_id="session-zero",
        session_dir=tmp_path,
        work_dir=tmp_path,
        review_md_path=tmp_path / "review.md",
        repo_dir=tmp_path,
        config_path=config_path,
    )

    # Synthesis runs on the raw artifacts; no terminal, no error status.
    assert synth == outputs
    assert skipped == "skipped"
    assert so_ctx is not None
    assert so_ctx.skipped is False
    assert so_ctx.matrix_rows == 0
    assert so_ctx.findings_in == 0
    assert meta.get("status") != "error"
    assert meta.get("multipass", {}).get("status") != "step_failed"
    assert playbook["calls"] == 0


def test_lenient_runtime_and_review_completion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import cure
    from cure_runtime import _doctor_system_one_config_check
    from cure_system_one import SystemOneClient

    outputs = _step_outputs(tmp_path, count=1)
    monkeypatch.setattr(
        cure, "_prepare_synth_inputs", lambda **kwargs: (list(outputs), "skipped")
    )

    calls: dict[str, int] = {"from_snapshot": 0}
    real_from_snapshot = SystemOneClient.from_snapshot

    def counting_from_snapshot(*args: Any, **kwargs: Any) -> Any:
        calls["from_snapshot"] += 1
        return real_from_snapshot(*args, **kwargs)

    monkeypatch.setattr(SystemOneClient, "from_snapshot", counting_from_snapshot)

    config_path = _write_config(
        tmp_path, '[system_one]\nenabled = true\nprovider = "bogus"\n'
    )
    meta: dict[str, Any] = {}
    synth, _skipped, so_ctx = cure._prepare_synth_inputs_with_system_one(
        meta=meta,
        step_entries=[],
        session_id="session-lenient",
        session_dir=tmp_path,
        work_dir=tmp_path,
        review_md_path=tmp_path / "review.md",
        repo_dir=tmp_path,
        config_path=config_path,
    )

    # Lenient at runtime: raw artifacts, one config degrade per skipped stage,
    # and zero provider calls.
    assert synth == outputs
    assert so_ctx is not None
    assert so_ctx.skipped is True
    assert [record["cause"] for record in so_ctx.degraded] == ["config", "config"]
    assert [record["kind"] for record in so_ctx.degraded] == ["config", "config"]
    assert calls["from_snapshot"] == 0

    # Strict at doctor time: the same section fails its config check (exit 1).
    check = _doctor_system_one_config_check({"enabled": True, "provider": "bogus"})
    assert check is not None
    assert check.status == "fail"


def test_advisory_emitted_once(tmp_path: Path, capsys: Any) -> None:
    import cure

    cure.clear_active_output()

    def system_one_meta(degraded: Any) -> dict[str, Any]:
        return {
            "multipass": {
                "system_one": {"degraded": degraded, "advisory_emitted": False}
            }
        }

    # A clean run prints nothing.
    clean = system_one_meta([])
    cure.emit_system_one_degraded_advisory(clean)
    assert capsys.readouterr().err == ""

    # One skipped stage prints exactly one stderr line naming the cause.
    single = system_one_meta([{"cause": "provider_failure", "kind": "transient"}])
    cure.emit_system_one_degraded_advisory(single)
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "provider_failure" in err
    assert single["multipass"]["system_one"]["advisory_emitted"] is True

    # The guard makes a second call a no-op.
    cure.emit_system_one_degraded_advisory(single)
    assert capsys.readouterr().err == ""

    # A chained skip names only the initiating (earliest-stage) cause once.
    chained = system_one_meta(
        [
            {"cause": "provider_failure", "kind": "transient"},
            {"cause": "config", "kind": "config"},
        ]
    )
    cure.emit_system_one_degraded_advisory(chained)
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.count("provider_failure") == 1
    assert "config" not in err

    # The advisory leaves review.md untouched.
    review = tmp_path / "review.md"
    review.write_text("frozen review body\n", encoding="utf-8")
    before = review.read_text(encoding="utf-8")
    cure.emit_system_one_degraded_advisory(system_one_meta([{"cause": "config"}]))
    capsys.readouterr()
    assert review.read_text(encoding="utf-8") == before


def test_meta_additive_records(tmp_path: Path, monkeypatch: Any) -> None:
    import cure
    import cure_sessions
    from cure_system_one.pipeline import SynthInputs
    from cure_system_one.preflight import FindingExclusion

    outputs = _step_outputs(tmp_path, count=2)
    monkeypatch.setattr(
        cure, "_prepare_synth_inputs", lambda **kwargs: (list(outputs), "skipped")
    )

    matrix_path = tmp_path / "system-one-evidence-matrix.json"
    matrix_path.write_text('{"rows": []}\n', encoding="utf-8")
    dedup = {
        "summary": {
            "findings_in": 3,
            "clusters_out": 2,
            "merged_count": 1,
            "passthrough_artifacts": 0,
            "budget_ms_consumed": 5,
            "provider": "local",
        },
        "clusters": [
            {
                "id": "c1",
                "provenance": [],
                "representative": "- [High] a",
                "confidence": [],
            }
        ],
        "passthrough": [],
        "degraded": None,
    }
    exclusion = FindingExclusion(
        cluster_id="c2",
        step_index=0,
        card_position=1,
        layer="mechanical",
        reason="citation_dead",
        preview="dead finding",
    )
    so_ctx = SynthInputs(
        matrix_path=str(matrix_path),
        matrix_rows=2,
        preflight_summary={
            "findings_in": 3,
            "citations_checked": 2,
            "mechanical_excluded": 1,
            "semantic_excluded": 0,
            "matrix_rows": 2,
            "budget_ms_consumed": 7,
            "provider": "local",
        },
        exclusions=(exclusion,),
        degraded=({"cause": "config", "kind": "config"},),
        dedup=dedup,
    )
    monkeypatch.setattr(
        cure, "resolve_synth_inputs", lambda artifacts, **kwargs: so_ctx
    )

    config_path = _write_config(
        tmp_path,
        '[system_one]\nenabled = true\nprovider = "local"\nmodel = "M6"\n',
    )
    meta: dict[str, Any] = {}
    cure._prepare_synth_inputs_with_system_one(
        meta=meta,
        step_entries=[],
        session_id="session-meta",
        session_dir=tmp_path,
        work_dir=tmp_path,
        review_md_path=tmp_path / "review.md",
        repo_dir=tmp_path,
        config_path=config_path,
    )

    system_one = meta["multipass"]["system_one"]
    assert system_one["status"] == {
        "enabled": True,
        "provider": "local",
        "model": "M6",
    }
    assert system_one["dedup"]["summary"]["clusters_out"] == 2
    assert system_one["dedup"]["clusters"][0]["id"] == "c1"
    assert system_one["preflight"]["summary"]["matrix_rows"] == 2
    assert system_one["preflight"]["exclusions"][0]["cluster_id"] == "c2"
    assert system_one["matrix_path"] == str(matrix_path)
    assert system_one["degraded"] == [{"cause": "config", "kind": "config"}]
    assert system_one["advisory_emitted"] is False

    # Existing session readers tolerate the additive records.
    assert cure_sessions._multipass_has_invalid_artifacts(meta) is False
    assert isinstance(cure_sessions.resolve_meta_llm(meta), dict)


# ---------------------------------------------------------------------------
# Known-bad replay corpus driven through the real multipass review harness.
# ---------------------------------------------------------------------------

_MATRIX_FILENAME = "system-one-evidence-matrix.json"
_DEAD_FINDING = "[High] Dead citation finding one."
_UNSUPPORTED_FINDING = "[High] Unsupported finding two."


def _replay_prose() -> str:
    sentence = "lorem ipsum dolor sit amet " * 6
    return "\n".join(f"- prose line {index}: {sentence}" for index in range(40))


def _replay_step_artifact(
    step_index: int, findings: list[str], *, include_findings: bool = True
) -> str:
    lines = [
        f"### Step Result: {step_index:02d} - Step {step_index}",
        "**Focus**: replay corpus",
        "",
        "### Steps taken",
        _replay_prose(),
        "",
        "### Hypothesis Ledger",
        "- suspicious surface: src/app.py; tentative issue: replay; next proof target: line 2",
        "",
    ]
    if include_findings:
        lines.extend(["### Findings", *findings, ""])
    lines.extend(["### Suggested actions", "- none", ""])
    return "\n".join(lines)


def _replay_corpus() -> dict[str, str]:
    step_one = [
        f"- {_DEAD_FINDING} Sources: `src/blank.py:5`",
        f"- {_UNSUPPORTED_FINDING} Sources: `src/app.py:1`",
        *(
            f"- [Medium] Valid finding {index}. Sources: `src/app.py:2`"
            for index in range(3, 7)
        ),
    ]
    step_two = [
        f"- [Medium] Valid finding {index}. Sources: `src/app.py:2`"
        for index in range(7, 13)
    ]
    return {
        "review.step-01.md": _replay_step_artifact(1, step_one),
        "review.step-02.md": _replay_step_artifact(2, step_two),
    }


def _replay_plan_text() -> str:
    return "\n".join(
        [
            "### Plan JSON",
            "```json",
            json.dumps(
                {
                    "abort": False,
                    "abort_reason": None,
                    "jira_keys": [],
                    "steps": [
                        {"id": "01", "title": "Step one", "focus": "replay"},
                        {"id": "02", "title": "Step two", "focus": "replay"},
                    ],
                }
            ),
            "```",
            "",
        ]
    )


def _replay_effective_paths(prompt: str) -> list[str]:
    marker = "- Step output files (read them all):"
    paths: list[str] = []
    for line in prompt[prompt.index(marker) + len(marker) :].splitlines():
        stripped = line.strip()
        if stripped.startswith("- `") and stripped.endswith("`"):
            paths.append(stripped[3:-1])
        elif paths:
            break
    return paths


def _synth_run_entries(meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in meta.get("multipass", {}).get("runs", [])
        if isinstance(entry, dict) and entry.get("kind") == "synth"
    ]


def _zero_findings_corpus() -> dict[str, str]:
    """Step artifacts that carry no ``### Findings`` section at all."""
    return {
        "review.step-01.md": _replay_step_artifact(1, [], include_findings=False),
        "review.step-02.md": _replay_step_artifact(2, [], include_findings=False),
    }


def _run_replay_flow(
    root: Path,
    *,
    flag_on: bool,
    corpus: dict[str, str] | None = None,
    grounding_mode: str = "strict",
) -> dict[str, Any]:
    """Drive the multipass review harness over the replay corpus.

    The corpus carries one mechanically-dead citation and one unsupported
    citation. The mocked synthesis reads only the effective
    ``STEP_OUTPUT_PATHS``, so a flag-on run synthesizes from the verified matrix
    while a flag-off run synthesizes from the raw step artifacts. ``corpus``
    overrides the default replay artifacts (e.g. for a finding-free review) and
    ``grounding_mode`` selects the multipass grounding mode.
    """
    import cure
    from cure_pr_context.corpus import estimated_tokens
    from _reviewflow_unittest_grounding_impl import (
        CodexToolProofFlowTests,
        MultipassGroundingRuntimeTests,
    )

    case = CodexToolProofFlowTests()
    grounding = MultipassGroundingRuntimeTests()
    corpus = corpus if corpus is not None else _replay_corpus()
    synth_state: dict[str, Any] = {"calls": 0, "effective_paths": [], "usage": {}}

    def clean_markdown() -> str:
        return grounding._valid_synth_markdown(primary_citation="src/app.py:2")

    def llm_side_effect(
        output_path: Path, work_dir: Path, kwargs: dict[str, Any]
    ) -> cure.LlmRunResult:
        name = output_path.name
        if name == "review.plan.md":
            output_path.write_text(_replay_plan_text(), encoding="utf-8")
            adapter_meta = case._write_helper_command_events(
                work_dir=work_dir, commands=["search", "research"]
            )
        elif name in corpus:
            repo_dir = Path(str(kwargs["repo_dir"]))
            (repo_dir / "src").mkdir(parents=True, exist_ok=True)
            (repo_dir / "src" / "app.py").write_text(
                "alpha\nbeta\ngamma\ndelta\n", encoding="utf-8"
            )
            (repo_dir / "src" / "blank.py").write_text(
                "one\ntwo\nthree\nfour\n\nsix\n", encoding="utf-8"
            )
            output_path.write_text(corpus[name], encoding="utf-8")
            adapter_meta = case._write_helper_command_events(
                work_dir=work_dir, commands=["search"]
            )
        elif name in {"review.md", "review.candidate.md"}:
            effective = _replay_effective_paths(str(kwargs["prompt"]))
            tokens = sum(
                estimated_tokens(Path(path).read_text(encoding="utf-8"))
                for path in effective
                if Path(path).is_file()
            )
            synth_state["calls"] += 1
            synth_state["effective_paths"] = effective
            synth_state["usage"][synth_state["calls"]] = tokens
            has_matrix = any(path.endswith(_MATRIX_FILENAME) for path in effective)
            if has_matrix or synth_state["calls"] > 1:
                output_path.write_text(clean_markdown(), encoding="utf-8")
            else:
                output_path.write_text(
                    clean_markdown().replace(
                        "- Missing provenance hygiene. Sources: `src/app.py:2`",
                        f"- {_DEAD_FINDING}",
                    ),
                    encoding="utf-8",
                )
            adapter_meta = case._write_codex_events(work_dir=work_dir, tool_names=[])
            adapter_meta["usage"] = {"input_tokens": tokens, "output_tokens": 1}
        else:
            raise AssertionError(f"unexpected output path: {output_path}")
        return cure.LlmRunResult(adapter_meta=adapter_meta)

    def flow_patch(stack: Any) -> None:
        if flag_on:
            config_path = root / "reviewflow.toml"
            config_path.write_text(
                config_path.read_text(encoding="utf-8")
                + '\n[system_one]\nenabled = true\nprovider = "local"\n',
                encoding="utf-8",
            )

    case._run_pr_flow_for_tool_proof(
        root=root,
        profile_resolved="big",
        multipass_enabled=True,
        llm_side_effect=llm_side_effect,
        multipass_defaults_override={
            "enabled": True,
            "max_steps": 20,
            "step_workers": 1,
            "grounding_mode": grounding_mode,
        },
        flow_patch=flow_patch,
    )
    session_dir = next(path for path in (root / "sandboxes").iterdir() if path.is_dir())
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "session_dir": session_dir,
        "meta": meta,
        "effective_paths": synth_state["effective_paths"],
        "usage": synth_state["usage"],
    }


@pytest.fixture(scope="module")
def replay_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    script_path = tmp_path_factory.mktemp("replay-script") / "system_one_script.json"
    script_path.write_text(
        json.dumps(
            {"noul": [{"value": {"probability": 0.4}}] + [{"value": {"probability": 0.9}}] * 20}
        ),
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(script_path))
        off = _run_replay_flow(tmp_path_factory.mktemp("replay-off"), flag_on=False)
        on = _run_replay_flow(tmp_path_factory.mktemp("replay-on"), flag_on=True)
    return {"off": off, "on": on}


@pytest.fixture(scope="module")
def zero_findings_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    script_path = (
        tmp_path_factory.mktemp("zero-findings-script") / "system_one_script.json"
    )
    script_path.write_text(
        json.dumps({"noul": [{"value": {"probability": 0.9}}] * 20}),
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("CURE_SYSTEM_ONE_SCRIPT", str(script_path))
        return _run_replay_flow(
            tmp_path_factory.mktemp("zero-findings"),
            flag_on=True,
            corpus=_zero_findings_corpus(),
            grounding_mode="off",
        )


def test_zero_findings_full_flow_runs_synthesis(
    zero_findings_run: dict[str, Any],
) -> None:
    # A flag-on review with no findings completes without the empty-matrix
    # terminal and still reaches synthesis.
    meta = zero_findings_run["meta"]
    assert meta.get("status") != "error"
    assert meta.get("multipass", {}).get("status") != "step_failed"
    system_one = meta["multipass"]["system_one"]
    assert system_one["preflight"]["summary"] is None
    assert system_one["matrix_path"] is None
    synth = _synth_run_entries(meta)
    assert synth, "synthesis must run on a finding-free review"
    assert (zero_findings_run["session_dir"] / "review.md").is_file()


def test_grounding_attempts_empty_on_known_bad_corpus(
    replay_runs: dict[str, Any],
) -> None:
    import cure

    off = replay_runs["off"]
    on = replay_runs["on"]

    # The known-bad corpus reaches pre-flight, which drops both bad citations.
    preflight = on["meta"]["multipass"]["system_one"]["preflight"]
    assert preflight["summary"]["findings_in"] == 12
    assert preflight["summary"]["matrix_rows"] == 10
    assert preflight["summary"]["mechanical_excluded"] == 1
    assert preflight["summary"]["semantic_excluded"] == 1
    assert {entry["reason"] for entry in preflight["exclusions"]} == {
        "citation_dead",
        "citation_unsupported",
    }
    assert (on["session_dir"] / _MATRIX_FILENAME).is_file()

    # Flag-off drops the bad findings through a grounding retry; flag-on is
    # verified before synthesis and never retries.
    off_synth = _synth_run_entries(off["meta"])
    on_synth = _synth_run_entries(on["meta"])
    assert len(off_synth) == 1 and len(on_synth) == 1
    assert len(off_synth[0]["grounding_attempts"]) == 1
    assert on_synth[0].get("grounding_attempts", []) == []
    assert on_synth[0].get("grounding_retried") is not True

    off_verdicts = cure.extract_review_verdicts_from_markdown(
        (off["session_dir"] / "review.md").read_text(encoding="utf-8")
    )
    on_verdicts = cure.extract_review_verdicts_from_markdown(
        (on["session_dir"] / "review.md").read_text(encoding="utf-8")
    )
    assert off_verdicts is not None and on_verdicts is not None
    assert on_verdicts.business == off_verdicts.business
    assert on_verdicts.technical == off_verdicts.technical


def test_token_ratio_ceiling(replay_runs: dict[str, Any]) -> None:
    off = replay_runs["off"]
    on = replay_runs["on"]

    off_synth = _synth_run_entries(off["meta"])
    on_synth = _synth_run_entries(on["meta"])
    assert set(off_synth[0]["usage"]) == {"input_tokens", "output_tokens", "total_tokens"}
    assert set(on_synth[0]["usage"]) == {"input_tokens", "output_tokens", "total_tokens"}

    # The matrix variant replaces the raw read list on the flag-on run.
    assert all(path.endswith(_MATRIX_FILENAME) for path in on["effective_paths"])
    assert all(path.endswith(".md") for path in off["effective_paths"])

    off_input = off_synth[0]["usage"]["input_tokens"]
    on_input = on_synth[0]["usage"]["input_tokens"]
    assert off_input > 0 and on_input > 0
    assert on_input / off_input <= 0.35


def test_flag_absent_suite_green(
    replay_runs: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    import cure

    off = replay_runs["off"]

    # No matrix artifact is produced or referenced on the raw path.
    assert not (off["session_dir"] / _MATRIX_FILENAME).exists()
    assert all("system-one-evidence-matrix" not in path for path in off["effective_paths"])
    assert off["effective_paths"]
    assert "system_one" not in off["meta"].get("multipass", {})

    # The disabled path resolves no System 1 context and leaves meta untouched.
    outputs = _step_outputs(tmp_path, count=1)
    monkeypatch.setattr(
        cure, "_prepare_synth_inputs", lambda **kwargs: (list(outputs), "skipped")
    )
    meta: dict[str, Any] = {}
    synth, skipped, so_ctx = cure._prepare_synth_inputs_with_system_one(
        meta=meta,
        step_entries=[],
        session_id="session-flag-absent",
        session_dir=tmp_path,
        work_dir=tmp_path,
        review_md_path=tmp_path / "review.md",
        repo_dir=tmp_path,
        config_path=_write_config(tmp_path, ""),
    )
    assert synth == outputs
    assert skipped == "skipped"
    assert so_ctx is None
    assert "system_one" not in meta.get("multipass", {})

