"""Unit tests for System 1 prompt-variable rendering."""
from __future__ import annotations


def test_render_preflight_omission_block_heading_and_entries() -> None:
    from cure_system_one.preflight import FindingExclusion
    from cure_system_one.prompt_vars import render_preflight_omission_block

    exclusions = [
        FindingExclusion(
            cluster_id="c1",
            step_index=0,
            card_position=0,
            layer="mechanical",
            reason="citation_dead",
            preview="dead finding",
        ),
        FindingExclusion(
            cluster_id="c2",
            step_index=1,
            card_position=2,
            layer="semantic",
            reason="citation_unsupported",
            preview="weak finding",
        ),
    ]
    block = render_preflight_omission_block(exclusions)
    lines = block.splitlines()
    assert lines[0] == "### Omitted Findings (System 1 Pre-flight)"
    assert len(lines) == 3
    assert "mechanical" in lines[1]
    assert "citation_dead" in lines[1]
    assert "dead finding" in lines[1]
    assert "semantic" in lines[2]
    assert "citation_unsupported" in lines[2]
    assert "weak finding" in lines[2]


def test_render_preflight_omission_block_empty() -> None:
    from cure_system_one.prompt_vars import render_preflight_omission_block

    assert (
        render_preflight_omission_block([])
        == "### Omitted Findings (System 1 Pre-flight)"
    )


def test_evidence_matrix_prompt_vars_off_and_on(tmp_path) -> None:
    from cure_system_one.pipeline import SynthInputs
    from cure_system_one.preflight import FindingExclusion
    from cure_system_one.prompt_vars import evidence_matrix_prompt_vars

    raw = "- `step-00.md`\n- `step-01.md`"

    # Off: no System 1 context, the raw list is returned untouched.
    off = evidence_matrix_prompt_vars(None, raw_step_paths_text=raw)
    assert off["EVIDENCE_MATRIX_GUIDANCE"] == ""
    assert off["EVIDENCE_MATRIX"] == ""
    assert off["STEP_OUTPUT_PATHS"] == raw

    # Off: a skipped stage selects the original raw path.
    skipped = evidence_matrix_prompt_vars(
        SynthInputs(skipped=True), raw_step_paths_text=raw
    )
    assert skipped["EVIDENCE_MATRIX_GUIDANCE"] == ""
    assert skipped["EVIDENCE_MATRIX"] == ""
    assert skipped["STEP_OUTPUT_PATHS"] == raw

    # Off: an empty matrix is treated as the original raw path.
    empty = evidence_matrix_prompt_vars(
        SynthInputs(matrix_path=None, matrix_rows=0), raw_step_paths_text=raw
    )
    assert empty["STEP_OUTPUT_PATHS"] == raw

    # On: guidance, matrix content, and the effective read list.
    matrix_path = tmp_path / "system-one-evidence-matrix.json"
    matrix_text = '{"rows": [{"cluster_id": "c1"}]}\n'
    matrix_path.write_text(matrix_text, encoding="utf-8")
    exclusion = FindingExclusion(
        cluster_id="c2",
        step_index=0,
        card_position=1,
        layer="semantic",
        reason="citation_unsupported",
        preview="weak finding",
    )
    so_ctx = SynthInputs(
        passthrough_paths=("step-02.md",),
        matrix_path=str(matrix_path),
        matrix_rows=1,
        preflight_summary={"matrix_rows": 1},
        exclusions=(exclusion,),
    )
    on = evidence_matrix_prompt_vars(so_ctx, raw_step_paths_text=raw)
    assert on["EVIDENCE_MATRIX"] == matrix_text
    assert on["EVIDENCE_MATRIX_GUIDANCE"]
    assert "### Omitted Findings (System 1 Pre-flight)" in on["EVIDENCE_MATRIX_GUIDANCE"]
    assert "citation_unsupported" in on["EVIDENCE_MATRIX_GUIDANCE"]
    assert "step-02.md" in on["EVIDENCE_MATRIX_GUIDANCE"]
    assert "verbatim" in on["EVIDENCE_MATRIX_GUIDANCE"].lower()
    paths = on["STEP_OUTPUT_PATHS"].splitlines()
    assert paths[0] == f"- `{matrix_path}`"
    assert paths[1] == "- `step-02.md`"
    assert raw not in on["STEP_OUTPUT_PATHS"]


def test_evidence_matrix_prompt_vars_missing_or_unreadable_artifact(tmp_path) -> None:
    from cure_system_one.pipeline import SynthInputs
    from cure_system_one.prompt_vars import evidence_matrix_prompt_vars

    raw = "- `step-00.md`"

    # A matrix path that no longer exists falls back to the off values.
    missing = SynthInputs(matrix_path=str(tmp_path / "gone.json"), matrix_rows=1)
    result = evidence_matrix_prompt_vars(missing, raw_step_paths_text=raw)
    assert result["EVIDENCE_MATRIX"] == ""
    assert result["EVIDENCE_MATRIX_GUIDANCE"] == ""
    assert result["STEP_OUTPUT_PATHS"] == raw

    # A path that is a directory (unreadable as a file) also falls back.
    directory = tmp_path / "as-dir"
    directory.mkdir()
    unreadable = SynthInputs(matrix_path=str(directory), matrix_rows=1)
    result = evidence_matrix_prompt_vars(unreadable, raw_step_paths_text=raw)
    assert result["EVIDENCE_MATRIX"] == ""
    assert result["STEP_OUTPUT_PATHS"] == raw
