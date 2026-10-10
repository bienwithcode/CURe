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
