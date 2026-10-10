"""Fence-aware parsing of agent finding cards from step artifacts.

Step artifacts are agent-produced markdown. This module segments the
``### Findings`` sections into individual finding cards, ignoring headings and
bullets that appear inside fenced code blocks, and preserves each bullet's text
byte-for-byte so downstream stages never rewrite finding content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from cure_citations import CITATION_LINE_RE, NONE_FINDINGS_BULLET, trailing_sources_suffix

SEVERITY_TOKENS = ("Critical", "High", "Medium", "Low", "Info")
UNSPECIFIED_SEVERITY = "unspecified"

_FINDINGS_TITLE = "Findings"
_SEVERITY_RE = re.compile(r"^\[(Critical|High|Medium|Low|Info)\]")
_FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})")


@dataclass(frozen=True)
class Citation:
    """One ``path:line`` citation parsed from a card's trailing ``Sources:``."""

    path: str
    line: int


@dataclass(frozen=True)
class FindingCard:
    """One verbatim finding bullet with its provenance and derived metadata."""

    step_index: int
    card_position: int
    text: str
    severity: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class ArtifactCards:
    """Result of parsing one artifact into finding cards.

    ``parse_failed`` distinguishes an artifact whose cards could not be trusted
    (an unterminated fenced code block) from a degenerate-valid artifact that
    simply carries no findings, so callers can pass the former through
    byte-identically instead of dropping it.
    """

    step_index: int
    cards: tuple[FindingCard, ...]
    parse_failed: bool = False
    failure_reason: str | None = None


def parse_findings(text: str, *, step_index: int = 0) -> ArtifactCards:
    """Parse ``text`` into finding cards, ignoring fenced code blocks.

    Bullets under a ``### Findings`` heading become cards in order, with
    ``card_position`` counting from zero across the whole artifact. A heading or
    bullet inside a fenced code block (``` or ~~~) is ignored. A fence left open
    at end of artifact marks the parse failed so the artifact can be passed
    through byte-identically rather than trusted partially.
    """
    cards: list[FindingCard] = []
    in_findings = False
    fence_char: str | None = None
    fence_length = 0
    position = 0
    for line in text.splitlines():
        if fence_char is not None:
            if _is_fence_close(line, fence_char, fence_length):
                fence_char = None
            continue
        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            fence_char = fence_match.group(2)[0]
            fence_length = len(fence_match.group(2))
            continue
        if line.startswith("### "):
            in_findings = line[4:].strip() == _FINDINGS_TITLE
            continue
        if line.startswith("## "):
            in_findings = False
            continue
        if in_findings and line.startswith("- "):
            if line.strip() == NONE_FINDINGS_BULLET:
                continue
            cards.append(
                FindingCard(
                    step_index=step_index,
                    card_position=position,
                    text=line,
                    severity=_severity_of(line),
                    citations=_citations_of(line),
                )
            )
            position += 1
    if fence_char is not None:
        return ArtifactCards(
            step_index=step_index,
            cards=tuple(cards),
            parse_failed=True,
            failure_reason="unterminated fenced code block",
        )
    return ArtifactCards(step_index=step_index, cards=tuple(cards))


def _is_fence_close(line: str, fence_char: str, fence_length: int) -> bool:
    match = re.match(
        r"^( {0,3})(" + re.escape(fence_char) + r"{" + str(fence_length) + r",})\s*$",
        line,
    )
    return match is not None


def _severity_of(line: str) -> str:
    match = _SEVERITY_RE.match(line[2:])
    return match.group(1) if match else UNSPECIFIED_SEVERITY


def _citations_of(line: str) -> tuple[Citation, ...]:
    suffix = trailing_sources_suffix(line)
    if not suffix:
        return ()
    return tuple(
        Citation(path=match.group(1), line=int(match.group(2)))
        for match in CITATION_LINE_RE.finditer(suffix)
    )
