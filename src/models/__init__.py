"""
Models package for Artifact Data Model.

Exports all Pydantic models, enums, and data structures for the
Trustworthy Model Registry Phase 2.
"""

from src.models.artifact import (
    ArtifactType,
    ArtifactBase,
    ArtifactMetadata,
    DownloadInfo,
    ArtifactCreate,
    ArtifactRead,
    ArtifactDB,
    to_dynamodb_item,
    from_dynamodb_item,
)
from src.models.metrics import (
    Phase1Metrics,
    Phase2Metrics,
    ArtifactMetrics,
)
from src.models.lineage import LineageInfo
from src.models.license import LicenseInfo, evaluate_license_compatibility

__all__ = [
    # Artifact models
    "ArtifactType",
    "ArtifactBase",
    "ArtifactMetadata",
    "DownloadInfo",
    "ArtifactCreate",
    "ArtifactRead",
    "ArtifactDB",
    "to_dynamodb_item",
    "from_dynamodb_item",
    # Metrics models
    "Phase1Metrics",
    "Phase2Metrics",
    "ArtifactMetrics",
    # Lineage models
    "LineageInfo",
    # License models
    "LicenseInfo",
    "evaluate_license_compatibility",
]
