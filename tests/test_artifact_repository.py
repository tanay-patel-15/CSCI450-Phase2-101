"""
Unit tests for ArtifactRepository.

Tests DynamoDB CRUD operations, GSI queries, and search
functionality using moto mock.
"""

import pytest
import boto3
from moto import mock_aws
from decimal import Decimal

from src.models.artifact import (
    ArtifactDB,
    ArtifactType,
    to_dynamodb_item,
    from_dynamodb_item,
)
from src.db.artifact_repository import (
    ArtifactRepository,
    ArtifactNotFoundError,
    ArtifactAlreadyExistsError,
    ArtifactRepositoryError,
    create_artifacts_table,
)


@pytest.fixture
def dynamodb_resource():
    """Create a mocked DynamoDB resource."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        yield dynamodb


@pytest.fixture
def artifacts_table(dynamodb_resource):
    """Create the artifacts table with GSIs."""
    client = boto3.client("dynamodb", region_name="us-east-1")
    
    client.create_table(
        TableName="test-artifacts",
        KeySchema=[
            {"AttributeName": "artifact_id", "KeyType": "HASH"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "artifact_id", "AttributeType": "S"},
            {"AttributeName": "name", "AttributeType": "S"},
            {"AttributeName": "type", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI_Name",
                "KeySchema": [{"AttributeName": "name", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI_Type",
                "KeySchema": [{"AttributeName": "type", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI_Name_Version",
                "KeySchema": [
                    {"AttributeName": "name", "KeyType": "HASH"},
                    {"AttributeName": "version", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    
    return dynamodb_resource.Table("test-artifacts")


@pytest.fixture
def repository(dynamodb_resource, artifacts_table):
    """Create an ArtifactRepository instance."""
    return ArtifactRepository(
        table_name="test-artifacts",
        dynamodb_resource=dynamodb_resource,
    )


@pytest.fixture
def sample_artifact():
    """Create a sample ArtifactDB for testing."""
    return ArtifactDB(
        artifact_id="test-artifact-123",
        name="test-model",
        type="MODEL",
        version="1.0.0",
        description="A test model",
        metadata={"size_bytes": 1000, "tags": ["test"]},
        lineage={"parents": [], "children": []},
        cost=0.5,
    )


class TestArtifactRepositoryCRUD:
    """Tests for basic CRUD operations."""
    
    @mock_aws
    def test_put_artifact(self, repository, sample_artifact):
        """Test creating an artifact."""
        result = repository.put_artifact(sample_artifact)
        
        assert result.artifact_id == sample_artifact.artifact_id
        assert result.name == sample_artifact.name
    
    @mock_aws
    def test_get_artifact(self, repository, sample_artifact):
        """Test retrieving an artifact."""
        repository.put_artifact(sample_artifact)
        
        result = repository.get_artifact(sample_artifact.artifact_id)
        
        assert result.artifact_id == sample_artifact.artifact_id
        assert result.name == sample_artifact.name
        assert result.type == "MODEL"
    
    @mock_aws
    def test_get_artifact_not_found(self, repository):
        """Test retrieving non-existent artifact."""
        with pytest.raises(ArtifactNotFoundError):
            repository.get_artifact("nonexistent-id")
    
    @mock_aws
    def test_delete_artifact(self, repository, sample_artifact):
        """Test deleting an artifact."""
        repository.put_artifact(sample_artifact)
        
        result = repository.delete_artifact(sample_artifact.artifact_id)
        
        assert result is True
        with pytest.raises(ArtifactNotFoundError):
            repository.get_artifact(sample_artifact.artifact_id)
    
    @mock_aws
    def test_delete_artifact_not_found(self, repository):
        """Test deleting non-existent artifact."""
        with pytest.raises(ArtifactNotFoundError):
            repository.delete_artifact("nonexistent-id")
    
    @mock_aws
    def test_put_artifact_no_overwrite(self, repository, sample_artifact):
        """Test creating artifact without overwrite."""
        repository.put_artifact(sample_artifact)
        
        with pytest.raises(ArtifactAlreadyExistsError):
            repository.put_artifact(sample_artifact, overwrite=False)


class TestArtifactRepositoryQueries:
    """Tests for GSI query operations."""
    
    @mock_aws
    def test_query_by_name(self, repository):
        """Test querying by name."""
        # Create multiple artifacts with same name
        a1 = ArtifactDB(artifact_id="id1", name="shared-name", type="MODEL", version="1.0.0")
        a2 = ArtifactDB(artifact_id="id2", name="shared-name", type="MODEL", version="2.0.0")
        a3 = ArtifactDB(artifact_id="id3", name="other-name", type="MODEL", version="1.0.0")
        
        repository.put_artifact(a1)
        repository.put_artifact(a2)
        repository.put_artifact(a3)
        
        results = repository.query_by_name("shared-name")
        
        assert len(results) == 2
        assert all(r.name == "shared-name" for r in results)
    
    @mock_aws
    def test_query_by_type(self, repository):
        """Test querying by artifact type."""
        m1 = ArtifactDB(artifact_id="m1", name="model1", type="MODEL", version="1.0.0")
        d1 = ArtifactDB(artifact_id="d1", name="dataset1", type="DATASET", version="1.0.0")
        c1 = ArtifactDB(artifact_id="c1", name="code1", type="CODE", version="1.0.0")
        
        repository.put_artifact(m1)
        repository.put_artifact(d1)
        repository.put_artifact(c1)
        
        models = repository.query_by_type(ArtifactType.MODEL)
        datasets = repository.query_by_type(ArtifactType.DATASET)
        
        assert len(models) == 1
        assert models[0].type == "MODEL"
        assert len(datasets) == 1
        assert datasets[0].type == "DATASET"
    
    @mock_aws
    def test_query_by_name_version(self, repository):
        """Test querying by name and version."""
        a1 = ArtifactDB(artifact_id="id1", name="model", type="MODEL", version="1.0.0")
        a2 = ArtifactDB(artifact_id="id2", name="model", type="MODEL", version="2.0.0")
        
        repository.put_artifact(a1)
        repository.put_artifact(a2)
        
        # Query specific version
        results = repository.query_by_name_version("model", "1.0.0")
        assert len(results) == 1
        assert results[0].version == "1.0.0"
        
        # Query all versions
        all_versions = repository.query_by_name_version("model")
        assert len(all_versions) == 2


class TestArtifactRepositoryScan:
    """Tests for scan operations."""
    
    @mock_aws
    def test_scan_all(self, repository):
        """Test scanning all artifacts."""
        for i in range(5):
            artifact = ArtifactDB(
                artifact_id=f"id{i}",
                name=f"artifact{i}",
                type="MODEL",
                version="1.0.0",
            )
            repository.put_artifact(artifact)
        
        results = repository.scan_all()
        
        assert len(results) == 5
    
    @mock_aws
    def test_scan_all_with_limit(self, repository):
        """Test scanning with limit."""
        for i in range(10):
            artifact = ArtifactDB(
                artifact_id=f"id{i}",
                name=f"artifact{i}",
                type="MODEL",
                version="1.0.0",
            )
            repository.put_artifact(artifact)
        
        results = repository.scan_all(limit=3)
        
        assert len(results) == 3
    
    @mock_aws
    def test_scan_with_regex_filter(self, repository):
        """Test scanning with regex filter."""
        repository.put_artifact(ArtifactDB(artifact_id="1", name="bert-base", type="MODEL", version="1.0.0"))
        repository.put_artifact(ArtifactDB(artifact_id="2", name="bert-large", type="MODEL", version="1.0.0"))
        repository.put_artifact(ArtifactDB(artifact_id="3", name="gpt2-small", type="MODEL", version="1.0.0"))
        
        results = repository.scan_with_regex_filter(r"bert-.*")
        
        assert len(results) == 2
        assert all("bert" in r.name for r in results)
    
    @mock_aws
    def test_scan_with_invalid_regex(self, repository):
        """Test that invalid regex raises ValueError."""
        with pytest.raises(ValueError):
            repository.scan_with_regex_filter("[invalid")


class TestArtifactRepositoryUpdates:
    """Tests for update operations."""
    
    @mock_aws
    def test_update_artifact_field(self, repository, sample_artifact):
        """Test updating a single field."""
        repository.put_artifact(sample_artifact)
        
        result = repository.update_artifact_field(
            sample_artifact.artifact_id,
            "description",
            "Updated description",
        )
        
        assert result.description == "Updated description"
    
    @mock_aws
    def test_update_lineage(self, repository, sample_artifact):
        """Test updating lineage."""
        repository.put_artifact(sample_artifact)
        
        result = repository.update_lineage(
            sample_artifact.artifact_id,
            parents=["parent-1", "parent-2"],
        )
        
        assert result.lineage["parents"] == ["parent-1", "parent-2"]
    
    @mock_aws
    def test_add_child_to_parent(self, repository, sample_artifact):
        """Test adding child reference to parent."""
        repository.put_artifact(sample_artifact)
        
        result = repository.add_child_to_parent(
            sample_artifact.artifact_id,
            "child-1",
        )
        
        assert "child-1" in result.lineage["children"]
    
    @mock_aws
    def test_increment_download_count(self, repository, sample_artifact):
        """Test incrementing download count."""
        sample_artifact.download_info = {"download_count": 0}
        repository.put_artifact(sample_artifact)
        
        result = repository.increment_download_count(sample_artifact.artifact_id)
        
        assert result.download_info["download_count"] == 1


class TestCreateArtifactsTable:
    """Tests for table creation helper."""
    
    @mock_aws
    def test_create_table(self):
        """Test creating the artifacts table."""
        client = boto3.client("dynamodb", region_name="us-east-1")
        
        create_artifacts_table("new-table", client)
        
        # Verify table exists
        tables = client.list_tables()["TableNames"]
        assert "new-table" in tables
    
    @mock_aws
    def test_create_table_already_exists(self):
        """Test creating table that already exists."""
        client = boto3.client("dynamodb", region_name="us-east-1")
        
        # Create twice - should not raise
        create_artifacts_table("existing-table", client)
        create_artifacts_table("existing-table", client)
