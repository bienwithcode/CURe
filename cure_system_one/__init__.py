"""System 1: schema-constrained provider decisions for CURe review synthesis.

The package is a leaf: it may import the leaf modules :mod:`cure_errors` and
:mod:`cure_citations` and must never import :mod:`cure`.
"""
from __future__ import annotations

from cure_system_one.cards import (
    ArtifactCards,
    Citation,
    FindingCard,
    parse_findings,
)
from cure_system_one.client import SystemOneClient, build_client
from cure_system_one.config import SystemOneConfig, resolve_system_one_config
from cure_system_one.dedup import (
    ClusterMember,
    DedupCluster,
    DedupDegraded,
    DedupResult,
    DedupSummary,
    StepArtifact,
    run_dedup,
)
from cure_system_one.errors import (
    SystemOneConfigError,
    SystemOneError,
    SystemOneInputError,
    SystemOneProviderAuthError,
    SystemOneSchemaViolationError,
    SystemOneTransientError,
)
from cure_system_one.pipeline import SynthInputs, resolve_synth_inputs
from cure_system_one.preflight import (
    CITATION_DEAD,
    CITATION_SUPPORTED,
    CITATION_UNSUPPORTED,
    DEFAULT_MATRIX_FILENAME,
    MECHANICAL_LAYER,
    SEMANTIC_LAYER,
    CitationEntry,
    FindingExclusion,
    MatrixRow,
    PreflightDegraded,
    PreflightResult,
    PreflightSummary,
    run_preflight,
)
from cure_system_one.prompt_vars import render_preflight_omission_block
from cure_system_one.providers import (
    ProviderAdapter,
    ProviderTransportError,
    create_provider,
    provider_names,
)

__all__ = [
    "ArtifactCards",
    "CITATION_DEAD",
    "CITATION_SUPPORTED",
    "CITATION_UNSUPPORTED",
    "Citation",
    "CitationEntry",
    "ClusterMember",
    "DEFAULT_MATRIX_FILENAME",
    "DedupCluster",
    "DedupDegraded",
    "DedupResult",
    "DedupSummary",
    "FindingCard",
    "FindingExclusion",
    "MECHANICAL_LAYER",
    "MatrixRow",
    "PreflightDegraded",
    "PreflightResult",
    "PreflightSummary",
    "ProviderAdapter",
    "ProviderTransportError",
    "SEMANTIC_LAYER",
    "StepArtifact",
    "SynthInputs",
    "SystemOneClient",
    "SystemOneConfig",
    "SystemOneConfigError",
    "SystemOneError",
    "SystemOneInputError",
    "SystemOneProviderAuthError",
    "SystemOneSchemaViolationError",
    "SystemOneTransientError",
    "build_client",
    "create_provider",
    "parse_findings",
    "provider_names",
    "render_preflight_omission_block",
    "resolve_synth_inputs",
    "resolve_system_one_config",
    "run_dedup",
    "run_preflight",
]
