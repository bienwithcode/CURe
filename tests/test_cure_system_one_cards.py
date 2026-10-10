"""Unit tests for the fence-aware finding-card parser."""
from __future__ import annotations


FENCE_AWARE_ARTIFACT = (
    "## Business / Product Assessment\n"
    "\n"
    "### Findings\n"
    "- [High] First finding. Sources: `src/a.py:10`\n"
    "- Second finding keeps trailing spaces  \n"
    "\n"
    "```markdown\n"
    "### Findings\n"
    "- [Critical] fenced bullet must be ignored\n"
    "```\n"
    "\n"
    "### Notes\n"
    "- not a finding\n"
    "\n"
    "~~~\n"
    "### Findings\n"
    "- [Critical] tilde fenced bullet must be ignored\n"
    "~~~\n"
    "\n"
    "### Findings\n"
    "- [Low] Later finding. Sources: `src/b.py:2`\n"
)


def test_parse_findings_fence_aware_and_verbatim() -> None:
    from cure_system_one.cards import parse_findings

    parsed = parse_findings(FENCE_AWARE_ARTIFACT, step_index=3)
    assert parsed.step_index == 3
    assert parsed.parse_failed is False
    assert [card.card_position for card in parsed.cards] == [0, 1, 2]
    assert [card.step_index for card in parsed.cards] == [3, 3, 3]
    assert parsed.cards[0].text == "- [High] First finding. Sources: `src/a.py:10`"
    assert parsed.cards[1].text == "- Second finding keeps trailing spaces  "
    assert parsed.cards[2].text == "- [Low] Later finding. Sources: `src/b.py:2`"

    no_findings = parse_findings("## Summary\n\n- not a finding\n")
    assert no_findings.cards == ()
    assert no_findings.parse_failed is False


def test_severity_sentinel_and_citation_parsing() -> None:
    from cure_system_one.cards import parse_findings

    artifact = (
        "### Findings\n"
        "- [Critical] c\n"
        "- [High] h\n"
        "- [Medium] m\n"
        "- [Low] l\n"
        "- [Info] i\n"
        "- [Blocker] b\n"
        "- [high] lower\n"
        "- plain finding\n"
        "- [High] with cites. Sources: `src/a.py:10`, `src/b.py:20`\n"
        "- [High] incomplete. Sources: `src/a.py`\n"
    )
    parsed = parse_findings(artifact)
    assert [card.severity for card in parsed.cards] == [
        "Critical",
        "High",
        "Medium",
        "Low",
        "Info",
        "unspecified",
        "unspecified",
        "unspecified",
        "High",
        "High",
    ]
    cited = parsed.cards[8]
    assert [(c.path, c.line) for c in cited.citations] == [
        ("src/a.py", 10),
        ("src/b.py", 20),
    ]
    assert parsed.cards[7].citations == ()
    assert parsed.cards[9].citations == ()


def test_parse_failure_marker() -> None:
    from cure_system_one.cards import parse_findings

    broken = "### Findings\n- [High] ok\n\n```\n- [High] stray unterminated\n"
    parsed = parse_findings(broken)
    assert parsed.parse_failed is True
    assert parsed.failure_reason

    empty = parse_findings("## Nothing to report\n")
    assert empty.parse_failed is False
    assert empty.cards == ()
    assert parsed.parse_failed != empty.parse_failed


def test_contract_placeholder_is_not_a_card() -> None:
    from cure_citations import NONE_FINDINGS_BULLET
    from cure_system_one.cards import parse_findings

    assert NONE_FINDINGS_BULLET == "- None."
    parsed = parse_findings("### Findings\n- None.\n")
    assert parsed.cards == ()
    assert parsed.parse_failed is False
