"""Flag-conditional synthesis prompt variables for the System 1 pipeline.

The pre-flight omission block is prompt guidance: it names every finding the
pre-flight stage excluded and why, so synthesis can disclose the omissions
without treating them as grounded evidence.
"""
from __future__ import annotations

from typing import Sequence

from cure_system_one.preflight import FindingExclusion

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
