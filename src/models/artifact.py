"""
Core Artifact schemas and models.

This module defines the central Artifact entity used across:
- DynamoDB storage
- API (Pydantic models)
- Internal business logic

All artifact types (MODEL, DATASET, CODE) share this base structure.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, List
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict
from .lineage import LineageInfo


class ArtifactType(str, Enum):
    """
    Enumeration of artifact types supported by the registry.
    
    Values:
        MODEL: Machine learning model artifacts
        DATASET: Training/evaluation dataset artifacts
        CODE: Code/script artifacts (training scripts, inference code, etc.)
    """
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"


class ArtifactMetadata(BaseModel):
    """
    Metadata fields for an artifact.
    
    Contains additional descriptive and technical information
    about the artifact beyond the core required fields.
    """
    model_config = ConfigDict(extra="forbid")
    
    size_bytes: int = Field(
        default=0,
        ge=0,
        description="S3 object byte size"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional user-supplied tags for categorization"
    )
    hf_model_id: Optional[str] = Field(
        default=None,
        description="Hugging Face model ID extracted during ingest"
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Languages in training corpus (e.g., 'en', 'fr')"
    )
    framework: str = Field(
        default="unknown",
        description="ML framework (e.g., 'pytorch', 'tensorflow', 'jax')"
    )
    task_type: str = Field(
        default="unknown",
        description="Task type (e.g., 'text-generation', 'image-classification')"
    )

    @field_validator("tags", "languages", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        """Ensure tags and languages are always lists."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class DownloadInfo(BaseModel):
    """
    Download-related information for an artifact.
    
    Tracks S3 storage location and download statistics.
    """
    model_config = ConfigDict(extra="forbid")
    
    s3_key: str = Field(
        default="",
        description="S3 object key for the artifact file"
    )
    download_url: str = Field(
        default="",
        description="Pre-signed or direct download URL"
    )
    download_count: int = Field(
        default=0,
        ge=0,
        description="Number of times artifact has been downloaded"
    )
    last_download_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last download"
    )


class ArtifactBase(BaseModel):
    """
    Base artifact model with shared fields.
    
    This is the foundation for all artifact-related schemas,
    containing the core required fields that every artifact must have.
    """
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable identifier for the artifact"
    )
    type: ArtifactType = Field(
        ...,
        description="Artifact class (MODEL, DATASET, or CODE)"
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?(\+[\w.]+)?$",
        description="Semantic version (e.g., '1.2.3', '1.0.0-beta.1')"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Detailed description (HF model card summary if available)"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Original external source URL (e.g., Hugging Face, GitHub)"
    )


class ArtifactCreate(ArtifactBase):
    """
    Schema for creating a new artifact (client POST body).
    
    Extends ArtifactBase with optional metadata fields that can
    be provided at creation time.
    """
    metadata: Optional[ArtifactMetadata] = Field(
        default_factory=ArtifactMetadata,
        description="Optional metadata for the artifact"
    )
    
    # CRITICAL FIX: Add the parent_artifact_ids field. 
    # This ensures the model includes the field that artifact_create_to_db relies on.
    parent_artifact_ids: Optional[List[str]] = Field(
        default=[],
        description="List of parent artifact IDs for lineage tracking."
    )


class ArtifactRead(ArtifactBase):
    """
    Full artifact read response schema.
    
    Includes all fields returned when fetching an artifact,
    including system-generated fields like ID and timestamps.
    """
    model_config = ConfigDict(from_attributes=True)
    
    artifact_id: str = Field(
        ...,
        description="Unique identifier (UUIDv4)"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when artifact was created"
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp of last modification"
    )
    metadata: ArtifactMetadata = Field(
        default_factory=ArtifactMetadata,
        description="Artifact metadata"
    )
    download_info: DownloadInfo = Field(
        default_factory=DownloadInfo,
        description="Download-related information"
    )
    # Nested imports to avoid circular dependencies
    metrics: Optional[dict[str, Any]] = Field(
        default=None,
        description="Artifact metrics (phase1 and phase2)"
    )
    lineage: Optional[LineageInfo] = Field(
        default=None,
        description="Lineage information (parents and children)"
    )
    license_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="License information"
    )
    cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Computed cost (artifact_size + sum(parent_costs))"
    )


class ArtifactDB(BaseModel):
    """
    DynamoDB storage model for artifacts.
    
    This model represents the full artifact record as stored
    in DynamoDB, including all nested structures serialized
    as dictionaries for DDB compatibility.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Primary key
    artifact_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Primary key (UUIDv4)"
    )
    
    # Core fields (for GSIs)
    name: str = Field(..., description="Human-readable identifier")
    type: str = Field(..., description="Artifact type string")
    version: str = Field(default="1.0.0", description="Semantic version")
    
    # Timestamps
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of creation"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of last update"
    )
    
    # Optional fields
    description: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    
    # Nested structures (stored as dicts in DynamoDB)
    metadata: dict[str, Any] = Field(default_factory=dict)
    download_info: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    license_info: dict[str, Any] = Field(default_factory=dict)
    cost: float = Field(default=0.0, ge=0.0)

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> str:
        """Convert ArtifactType enum to string if needed."""
        if isinstance(v, ArtifactType):
            return v.value
        return str(v).upper()


def _convert_decimals(obj: Any) -> Any:
    """
    Recursively convert Decimal values to float/int for JSON serialization.
    
    DynamoDB returns numeric values as Decimal, which need to be
    converted for JSON serialization and Pydantic model usage.
    """
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals(i) for i in obj]
    return obj


def _convert_floats_to_decimal(obj: Any) -> Any:
    """
    Recursively convert float values to Decimal for DynamoDB storage.
    
    DynamoDB requires Decimal for numeric values instead of float.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimal(i) for i in obj]
    return obj


def to_dynamodb_item(artifact: ArtifactDB) -> dict[str, Any]:
    """
    Serialize an ArtifactDB model to a DynamoDB item.
    
    Converts the Pydantic model to a dictionary suitable for
    DynamoDB put_item operations, handling type conversions.
    
    Args:
        artifact: The ArtifactDB model to serialize
        
    Returns:
        Dictionary ready for DynamoDB put_item
    """
    item = artifact.model_dump(exclude_none=True)
    # Convert floats to Decimal for DynamoDB
    item = _convert_floats_to_decimal(item)
    return item


def from_dynamodb_item(item: dict[str, Any]) -> ArtifactDB:
    """
    Deserialize a DynamoDB item to an ArtifactDB model.
    
    Converts a DynamoDB item dictionary back to a Pydantic
    model, handling Decimal to float/int conversions.
    
    Args:
        item: DynamoDB item dictionary
        
    Returns:
        ArtifactDB model instance
    """
    # Convert Decimal values from DynamoDB
    converted = _convert_decimals(item)
    return ArtifactDB(**converted)


def artifact_db_to_read(artifact_db: ArtifactDB) -> ArtifactRead:
    """
    Convert an ArtifactDB model to an ArtifactRead response model.
    
    Args:
        artifact_db: The database model
        
    Returns:
        ArtifactRead model for API responses
    """
    return ArtifactRead(
        artifact_id=artifact_db.artifact_id,
        name=artifact_db.name,
        type=ArtifactType(artifact_db.type),
        version=artifact_db.version,
        description=artifact_db.description,
        source_url=artifact_db.source_url,
        created_at=datetime.fromisoformat(artifact_db.created_at),
        updated_at=datetime.fromisoformat(artifact_db.updated_at),
        metadata=ArtifactMetadata(**artifact_db.metadata) if artifact_db.metadata else ArtifactMetadata(),
        download_info=DownloadInfo(**artifact_db.download_info) if artifact_db.download_info else DownloadInfo(),
        metrics=artifact_db.metrics,
        lineage=LineageInfo.from_dict(artifact_db.lineage) if artifact_db.lineage else None,
        license_info=artifact_db.license_info,
        cost=artifact_db.cost,
    )


def artifact_create_to_db(artifact_create: ArtifactCreate) -> ArtifactDB:
    """
    Convert an ArtifactCreate model to an ArtifactDB model.
    
    Generates new artifact_id and timestamps for the new artifact.
    
    Args:
        artifact_create: The creation request model
        
    Returns:
        ArtifactDB model ready for storage
    """
    now = datetime.utcnow().isoformat()
    metadata_dict = artifact_create.metadata.model_dump() if artifact_create.metadata else {}
    
    return ArtifactDB(
        artifact_id=str(uuid4()),
        name=artifact_create.name,
        type=artifact_create.type.value,
        version=artifact_create.version,
        description=artifact_create.description,
        source_url=artifact_create.source_url,
        created_at=now,
        updated_at=now,
        metadata=metadata_dict,
        download_info={},
        metrics={"phase1": {}, "phase2": {}},
        lineage={"parents": artifact_create.parent_artifact_ids, "children": []},
        license_info={},
        cost=0.0,
    )
