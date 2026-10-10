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
    passthrough path.
    """
    if so_ctx is None or so_ctx.skipped or so_ctx.matrix_path is None:
        return {
            "EVIDENCE_MATRIX_GUIDANCE": "",
            "EVIDENCE_MATRIX": "",
            "STEP_OUTPUT_PATHS": raw_step_paths_text,
        }
    matrix_text = Path(so_ctx.matrix_path).read_text(encoding="utf-8")
    effective_paths = (so_ctx.matrix_path, *so_ctx.passthrough_paths)
    return {
        "EVIDENCE_MATRIX_GUIDANCE": _render_matrix_guidance(so_ctx),
        "EVIDENCE_MATRIX": matrix_text,
        "STEP_OUTPUT_PATHS": "\n".join(f"- `{path}`" for path in effective_paths),
    }


def _render_matrix_guidance(so_ctx: "SynthInputs") -> str:
    """Render the guidance that accompanies a present evidence matrix."""
    lines = [
        "System 1 pre-flight produced a verified evidence matrix for this review.",
        "- The evidence matrix is authoritative: its findings already passed mechanical "
        "citation checks and semantic grounding, so carry them forward without re-verifying.",
        "- Read the evidence matrix artifact and synthesize from the findings it carries.",
        "- The evidence matrix already folds duplicate step findings and drops non-finding prose.",
    ]
    if so_ctx.passthrough_paths:
        lines.append(
            "- Read each passthrough artifact below verbatim: it could not be parsed into "
            "the matrix and must be considered in full."
        )
        for path in so_ctx.passthrough_paths:
            lines.append(f"- passthrough (read verbatim): `{path}`")
    omission_block = render_preflight_omission_block(so_ctx.exclusions)
    return "\n".join([*lines, "", omission_block])
