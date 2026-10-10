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
from cure_system_one.providers import (
    ProviderAdapter,
    ProviderTransportError,
    create_provider,
    provider_names,
)

__all__ = [
    "ArtifactCards",
    "Citation",
    "ClusterMember",
    "DedupCluster",
    "DedupDegraded",
    "DedupResult",
    "DedupSummary",
    "FindingCard",
    "ProviderAdapter",
    "ProviderTransportError",
    "StepArtifact",
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
    "resolve_system_one_config",
    "run_dedup",
]
