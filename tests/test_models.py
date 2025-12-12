"""
Unit tests for Pydantic models.

Tests artifact creation, validation, serialization, and
model conversion functions.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from models.artifact import (
    ArtifactType,
    ArtifactBase,
    ArtifactMetadata,
    DownloadInfo,
    ArtifactCreate,
    ArtifactRead,
    ArtifactDB,
    to_dynamodb_item,
    from_dynamodb_item,
    artifact_db_to_read,
    artifact_create_to_db,
)
from models.metrics import (
    Phase1Metrics,
    Phase2Metrics,
    ArtifactMetrics,
    SizeScore,
)
from models.lineage import LineageInfo
from models.license import (
    LicenseInfo,
    evaluate_license_compatibility,
    check_license_chain_compatibility,
)


class TestArtifactType:
    """Tests for ArtifactType enum."""
    
    def test_artifact_type_values(self):
        """Test that all artifact types have correct values."""
        assert ArtifactType.MODEL.value == "MODEL"
        assert ArtifactType.DATASET.value == "DATASET"
        assert ArtifactType.CODE.value == "CODE"
    
    def test_artifact_type_from_string(self):
        """Test creating ArtifactType from string."""
        assert ArtifactType("MODEL") == ArtifactType.MODEL
        assert ArtifactType("DATASET") == ArtifactType.DATASET
        assert ArtifactType("CODE") == ArtifactType.CODE
    
    def test_artifact_type_invalid(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError):
            ArtifactType("INVALID")


class TestArtifactMetadata:
    """Tests for ArtifactMetadata model."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        metadata = ArtifactMetadata()
        assert metadata.size_bytes == 0
        assert metadata.tags == []
        assert metadata.hf_model_id is None
        assert metadata.languages == []
        assert metadata.framework == "unknown"
        assert metadata.task_type == "unknown"
    
    def test_custom_values(self):
        """Test setting custom values."""
        metadata = ArtifactMetadata(
            size_bytes=1000000,
            tags=["nlp", "transformer"],
            hf_model_id="bert-base-uncased",
            languages=["en", "de"],
            framework="pytorch",
            task_type="text-classification",
        )
        assert metadata.size_bytes == 1000000
        assert metadata.tags == ["nlp", "transformer"]
        assert metadata.hf_model_id == "bert-base-uncased"
        assert metadata.languages == ["en", "de"]
        assert metadata.framework == "pytorch"
        assert metadata.task_type == "text-classification"
    
    def test_tags_from_string(self):
        """Test that single string tag is converted to list."""
        metadata = ArtifactMetadata(tags="single-tag")
        assert metadata.tags == ["single-tag"]
    
    def test_size_bytes_validation(self):
        """Test that negative size_bytes raises error."""
        with pytest.raises(ValueError):
            ArtifactMetadata(size_bytes=-1)


class TestArtifactBase:
    """Tests for ArtifactBase model."""
    
    def test_required_fields(self):
        """Test that name and type are required."""
        with pytest.raises(ValueError):
            ArtifactBase()
    
    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        artifact = ArtifactBase(name="test-model", type=ArtifactType.MODEL)
        assert artifact.name == "test-model"
        assert artifact.type == ArtifactType.MODEL
        assert artifact.version == "1.0.0"
        assert artifact.description is None
        assert artifact.source_url is None
    
    def test_version_validation(self):
        """Test semantic version validation."""
        # Valid versions
        ArtifactBase(name="test", type=ArtifactType.MODEL, version="1.0.0")
        ArtifactBase(name="test", type=ArtifactType.MODEL, version="1.0.0-beta.1")
        ArtifactBase(name="test", type=ArtifactType.MODEL, version="1.0.0+build.123")
        
        # Invalid version
        with pytest.raises(ValueError):
            ArtifactBase(name="test", type=ArtifactType.MODEL, version="invalid")


class TestArtifactCreate:
    """Tests for ArtifactCreate model."""
    
    def test_create_with_metadata(self):
        """Test creating artifact with metadata."""
        artifact = ArtifactCreate(
            name="test-model",
            type=ArtifactType.MODEL,
            metadata=ArtifactMetadata(size_bytes=5000),
        )
        assert artifact.metadata.size_bytes == 5000
    
    def test_default_metadata(self):
        """Test that metadata defaults to empty ArtifactMetadata."""
        artifact = ArtifactCreate(name="test", type=ArtifactType.DATASET)
        assert artifact.metadata is not None
        assert artifact.metadata.size_bytes == 0


class TestArtifactDB:
    """Tests for ArtifactDB model."""
    
    def test_auto_generated_fields(self):
        """Test that artifact_id and timestamps are auto-generated."""
        artifact = ArtifactDB(name="test", type="MODEL")
        assert artifact.artifact_id is not None
        assert len(artifact.artifact_id) == 36  # UUID format
        assert artifact.created_at is not None
        assert artifact.updated_at is not None
    
    def test_type_normalization(self):
        """Test that type is normalized to uppercase."""
        artifact = ArtifactDB(name="test", type=ArtifactType.MODEL)
        assert artifact.type == "MODEL"


class TestDynamoDBSerialization:
    """Tests for DynamoDB serialization/deserialization."""
    
    def test_to_dynamodb_item(self):
        """Test converting ArtifactDB to DynamoDB item."""
        artifact = ArtifactDB(
            artifact_id="test-id",
            name="test-model",
            type="MODEL",
            version="1.0.0",
            cost=0.5,
        )
        item = to_dynamodb_item(artifact)
        
        assert item["artifact_id"] == "test-id"
        assert item["name"] == "test-model"
        assert item["type"] == "MODEL"
        # Cost should be converted to Decimal
        from decimal import Decimal
        assert item["cost"] == Decimal("0.5")
    
    def test_from_dynamodb_item(self):
        """Test converting DynamoDB item to ArtifactDB."""
        from decimal import Decimal
        item = {
            "artifact_id": "test-id",
            "name": "test-model",
            "type": "MODEL",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "cost": Decimal("0.5"),
            "metadata": {"size_bytes": Decimal(1000)},
        }
        artifact = from_dynamodb_item(item)
        
        assert artifact.artifact_id == "test-id"
        assert artifact.name == "test-model"
        assert artifact.cost == 0.5
        assert artifact.metadata["size_bytes"] == 1000
    
    def test_roundtrip_serialization(self):
        """Test that serialization is reversible."""
        original = ArtifactDB(
            name="test",
            type="MODEL",
            cost=1.5,
            metadata={"size_bytes": 1000, "tags": ["a", "b"]},
        )
        item = to_dynamodb_item(original)
        restored = from_dynamodb_item(item)
        
        assert restored.name == original.name
        assert restored.type == original.type
        assert restored.cost == original.cost


class TestArtifactConversion:
    """Tests for artifact model conversion functions."""
    
    def test_artifact_create_to_db(self):
        """Test converting ArtifactCreate to ArtifactDB."""
        create = ArtifactCreate(
            name="test-model",
            type=ArtifactType.MODEL,
            version="2.0.0",
            description="Test description",
        )
        db = artifact_create_to_db(create)
        
        assert db.name == "test-model"
        assert db.type == "MODEL"
        assert db.version == "2.0.0"
        assert db.description == "Test description"
        assert db.artifact_id is not None
        assert db.metrics == {"phase1": {}, "phase2": {}}
        assert db.lineage == {"parents": [], "children": []}
    
    def test_artifact_db_to_read(self):
        """Test converting ArtifactDB to ArtifactRead."""
        db = ArtifactDB(
            artifact_id="test-id",
            name="test-model",
            type="MODEL",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        read = artifact_db_to_read(db)
        
        assert read.artifact_id == "test-id"
        assert read.name == "test-model"
        assert read.type == ArtifactType.MODEL
        assert isinstance(read.created_at, datetime)


class TestPhase1Metrics:
    """Tests for Phase1Metrics model."""
    
    def test_default_values(self):
        """Test all default values are zero."""
        metrics = Phase1Metrics()
        assert metrics.ramp_up_time == 0.0
        assert metrics.bus_factor == 0.0
        assert metrics.license_score == 0.0
        assert metrics.net_score == 0.0
    
    def test_size_score_default(self):
        """Test size_score has all hardware targets."""
        metrics = Phase1Metrics()
        assert "raspberry_pi" in metrics.size_score
        assert "jetson_nano" in metrics.size_score
        assert "desktop_pc" in metrics.size_score
        assert "aws_server" in metrics.size_score
    
    def test_value_bounds(self):
        """Test that values outside 0-1 are rejected."""
        with pytest.raises(ValueError):
            Phase1Metrics(ramp_up_time=1.5)
        with pytest.raises(ValueError):
            Phase1Metrics(bus_factor=-0.1)


class TestPhase2Metrics:
    """Tests for Phase2Metrics model."""
    
    def test_default_values(self):
        """Test default values."""
        metrics = Phase2Metrics()
        assert metrics.reproducibility == 0.0
        assert metrics.reviewedness == 0.0
        assert metrics.treescale_score == 0.0
        assert metrics.latency_ms == 0
    
    def test_reproducibility_validation(self):
        """Test reproducibility is clamped to allowed values."""
        # Should be clamped to 0.0, 0.5, or 1.0
        m1 = Phase2Metrics(reproducibility=0.2)
        assert m1.reproducibility == 0.0
        
        m2 = Phase2Metrics(reproducibility=0.6)
        assert m2.reproducibility == 0.5
        
        m3 = Phase2Metrics(reproducibility=0.9)
        assert m3.reproducibility == 1.0


class TestArtifactMetrics:
    """Tests for ArtifactMetrics combined model."""
    
    def test_nested_structure(self):
        """Test nested phase1/phase2 structure."""
        metrics = ArtifactMetrics()
        assert isinstance(metrics.phase1, Phase1Metrics)
        assert isinstance(metrics.phase2, Phase2Metrics)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ArtifactMetrics(
            phase1=Phase1Metrics(ramp_up_time=0.5),
            phase2=Phase2Metrics(latency_ms=100),
        )
        d = metrics.to_dict()
        assert d["phase1"]["ramp_up_time"] == 0.5
        assert d["phase2"]["latency_ms"] == 100
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "phase1": {"ramp_up_time": 0.7},
            "phase2": {"latency_ms": 50},
        }
        metrics = ArtifactMetrics.from_dict(data)
        assert metrics.phase1.ramp_up_time == 0.7
        assert metrics.phase2.latency_ms == 50


class TestLineageInfo:
    """Tests for LineageInfo model."""
    
    def test_default_empty(self):
        """Test default values are empty lists."""
        lineage = LineageInfo()
        assert lineage.parents == []
        assert lineage.children == []
    
    def test_add_parent(self):
        """Test adding a parent."""
        lineage = LineageInfo()
        assert lineage.add_parent("parent-1") is True
        assert "parent-1" in lineage.parents
        # Adding same parent again should return False
        assert lineage.add_parent("parent-1") is False
    
    def test_add_child(self):
        """Test adding a child."""
        lineage = LineageInfo()
        assert lineage.add_child("child-1") is True
        assert "child-1" in lineage.children
    
    def test_remove_child(self):
        """Test removing a child."""
        lineage = LineageInfo(children=["child-1", "child-2"])
        assert lineage.remove_child("child-1") is True
        assert "child-1" not in lineage.children
        assert lineage.remove_child("nonexistent") is False
    
    def test_is_root_and_leaf(self):
        """Test root and leaf detection."""
        root = LineageInfo(children=["child-1"])
        assert root.is_root is True
        assert root.is_leaf is False
        
        leaf = LineageInfo(parents=["parent-1"])
        assert leaf.is_root is False
        assert leaf.is_leaf is True


class TestLicenseInfo:
    """Tests for LicenseInfo model."""
    
    def test_default_values(self):
        """Test default license values."""
        license_info = LicenseInfo()
        assert license_info.license_id == "UNKNOWN"
        assert license_info.license_source == "unknown"
        assert license_info.license_compatible is False
    
    def test_source_normalization(self):
        """Test license source normalization."""
        assert LicenseInfo(license_source="huggingface").license_source == "hf"
        assert LicenseInfo(license_source="GitHub").license_source == "github"
        assert LicenseInfo(license_source="random").license_source == "unknown"
    
    def test_is_permissive(self):
        """Test permissive license detection."""
        mit = LicenseInfo(license_id="MIT")
        assert mit.is_permissive is True
        
        gpl = LicenseInfo(license_id="GPL-3.0-only")
        assert gpl.is_permissive is False
    
    def test_is_copyleft(self):
        """Test copyleft license detection."""
        gpl = LicenseInfo(license_id="GPL-3.0-only")
        assert gpl.is_copyleft is True
        
        mit = LicenseInfo(license_id="MIT")
        assert mit.is_copyleft is False


class TestLicenseCompatibility:
    """Tests for license compatibility functions."""
    
    def test_permissive_always_compatible(self):
        """Test permissive licenses are always compatible."""
        mit = LicenseInfo(license_id="MIT")
        assert evaluate_license_compatibility(mit) is True
    
    def test_unknown_not_compatible(self):
        """Test unknown licenses are not compatible."""
        unknown = LicenseInfo(license_id="UNKNOWN")
        assert evaluate_license_compatibility(unknown) is False
    
    def test_copyleft_with_flag(self):
        """Test copyleft licenses with allow_copyleft flag."""
        gpl = LicenseInfo(license_id="GPL-3.0-only")
        assert evaluate_license_compatibility(gpl, allow_copyleft=False) is False
        assert evaluate_license_compatibility(gpl, allow_copyleft=True) is True
