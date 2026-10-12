"""Flag-conditional synthesis prompt variables for the System 1 pipeline.

The pre-flight omission block is prompt guidance: it names every finding the
pre-flight stage excluded and why, so synthesis can disclose the omissions
without treating them as grounded evidence. ``evidence_matrix_prompt_vars``
selects between the raw step-output read list and the verified evidence matrix
variant.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from cure_system_one.preflight import FindingExclusion

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cure_system_one.pipeline import SynthInputs

OMISSION_BLOCK_HEADING = "### Omitted Findings (System 1 Pre-flight)"
EVIDENCE_MATRIX_HEADER = "- Evidence matrix (System 1 pre-flight; authoritative when present):"


def render_preflight_omission_block(
    exclusions: Sequence[FindingExclusion],
) -> str:
    """Render the pre-flight omission guidance block.

    The block always starts with the exact heading and then carries one entry
    per exclusion naming its layer, reason, and preview. With no exclusions the
    heading renders alone.
    """
    lines = [OMISSION_BLOCK_HEADING]
    for exclusion in exclusions:
        lines.append(
            f"- [{exclusion.layer}] {exclusion.reason}: {exclusion.preview}"
        )
    return "\n".join(lines)


def evidence_matrix_prompt_vars(
    so_ctx: "SynthInputs | None",
    *,
    raw_step_paths_text: str,
) -> dict[str, str]:
    """Resolve the synthesis prompt's evidence-matrix variant.

    When System 1 is absent, skipped, or produced no matrix the guidance and
    matrix are empty and ``STEP_OUTPUT_PATHS`` is the raw list verbatim. When
    the matrix is present the guidance carries the pre-flight omission block and
    marks every passthrough artifact as a verbatim read, the matrix artifact is
    injected, and ``STEP_OUTPUT_PATHS`` lists the matrix path followed by every
    passthrough path. A matrix artifact that has gone missing or is unreadable
    falls back to the off values rather than failing synthesis. When pre-flight
    excluded every finding but passthrough artifacts remain, ``STEP_OUTPUT_PATHS``
    lists only the passthrough paths and the guidance carries the omission
    block, so excluded findings never re-enter synthesis through the raw list.

    The synth templates place both placeholders inline right after
    ``$STEP_OUTPUT_PATHS``, so the on-values carry their own leading blank lines
    and the matrix header. The off-values are empty strings, which renders the
    pre-feature prompt layout byte-for-byte.
    """
    if so_ctx is None or so_ctx.skipped:
        return _off_prompt_vars(raw_step_paths_text)
    if so_ctx.matrix_path is None:
        if so_ctx.findings_in > 0 and so_ctx.passthrough_paths:
            return _passthrough_only_prompt_vars(so_ctx)
        return _off_prompt_vars(raw_step_paths_text)
    try:
        matrix_text = Path(so_ctx.matrix_path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return _off_prompt_vars(raw_step_paths_text)
    effective_paths = (so_ctx.matrix_path, *so_ctx.passthrough_paths)
    matrix_body = matrix_text.rstrip("\n")
    return {
        "EVIDENCE_MATRIX_GUIDANCE": "\n\n" + _render_matrix_guidance(so_ctx),
        "EVIDENCE_MATRIX": f"\n\n{EVIDENCE_MATRIX_HEADER}\n{matrix_body}\n",
        "STEP_OUTPUT_PATHS": "\n".join(f"- `{path}`" for path in effective_paths),
    }


def _off_prompt_vars(raw_step_paths_text: str) -> dict[str, str]:
    """Return the raw-artifact variant shared by every off/skip path."""
    return {
        "EVIDENCE_MATRIX_GUIDANCE": "",
        "EVIDENCE_MATRIX": "",
        "STEP_OUTPUT_PATHS": raw_step_paths_text,
    }


def _passthrough_only_prompt_vars(so_ctx: "SynthInputs") -> dict[str, str]:
    """Return the variant for an all-excluded matrix with passthrough artifacts."""
    lines = [
        "System 1 pre-flight excluded every parsed finding for this review; only the "
        "passthrough artifacts below remain as synthesis input.",
        *_passthrough_guidance_lines(so_ctx),
        "",
        render_preflight_omission_block(so_ctx.exclusions),
    ]
    return {
        "EVIDENCE_MATRIX_GUIDANCE": "\n\n" + "\n".join(lines),
        "EVIDENCE_MATRIX": "",
        "STEP_OUTPUT_PATHS": "\n".join(
            f"- `{path}`" for path in so_ctx.passthrough_paths
        ),
    }


def _passthrough_guidance_lines(so_ctx: "SynthInputs") -> list[str]:
    """Render the verbatim-read guidance for every passthrough artifact."""
    if not so_ctx.passthrough_paths:
        return []
    lines = [
        "- Read each passthrough artifact below verbatim: it could not be parsed into "
        "the matrix and must be considered in full."
    ]
    for path in so_ctx.passthrough_paths:
        lines.append(f"- passthrough (read verbatim): `{path}`")
    return lines


def _render_matrix_guidance(so_ctx: "SynthInputs") -> str:
    """Render the guidance that accompanies a present evidence matrix."""
    lines = [
        "System 1 pre-flight produced a verified evidence matrix for this review.",
        "- The evidence matrix is authoritative: its findings already passed mechanical "
        "citation checks and semantic grounding, so carry them forward without re-verifying.",
        "- Read the evidence matrix artifact and synthesize from the findings it carries.",
        "- Cite the repository `path:line` carried in each row's `citations`; never cite "
        "the evidence matrix artifact itself as a source.",
        "- The evidence matrix already folds duplicate step findings and drops non-finding prose.",
    ]
    lines.extend(_passthrough_guidance_lines(so_ctx))
    omission_block = render_preflight_omission_block(so_ctx.exclusions)
    return "\n".join([*lines, "", omission_block])
