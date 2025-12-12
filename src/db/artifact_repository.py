"""
DynamoDB Artifact Repository.

This module provides the data access layer for artifact storage
in DynamoDB, including CRUD operations, GSI queries, and
search functionality.

Table Schema:
- Partition Key: artifact_id (String)
- No Sort Key

Global Secondary Indexes (GSIs):
- GSI_Name: name (PK) - for exact name lookup
- GSI_Type: type (PK) - for filtering by artifact type
- GSI_Name_Version: name (PK) + version (SK) - for semantic versioning queries
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from src.models.artifact import (
    ArtifactDB,
    ArtifactType,
    from_dynamodb_item,
    to_dynamodb_item,
)

logger = logging.getLogger(__name__)


class ArtifactRepositoryError(Exception):
    """Base exception for artifact repository errors."""
    pass


class ArtifactNotFoundError(ArtifactRepositoryError):
    """Raised when an artifact is not found."""
    pass


class ArtifactAlreadyExistsError(ArtifactRepositoryError):
    """Raised when trying to create an artifact that already exists."""
    pass


class ArtifactRepository:
    """
    DynamoDB repository for artifact storage and retrieval.
    
    Provides CRUD operations, GSI queries, and search functionality
    for artifacts stored in DynamoDB.
    
    Attributes:
        table_name: Name of the DynamoDB table
        table: boto3 DynamoDB Table resource
    """
    
    # GSI names
    GSI_NAME = "GSI_Name"
    GSI_TYPE = "GSI_Type"
    GSI_NAME_VERSION = "GSI_Name_Version"
    
    def __init__(
        self,
        table_name: Optional[str] = None,
        dynamodb_resource: Optional[Any] = None,
    ):
        """
        Initialize the artifact repository.
        
        Args:
            table_name: DynamoDB table name. Defaults to ARTIFACTS_TABLE env var or 'artifacts'.
            dynamodb_resource: Optional boto3 DynamoDB resource for testing.
        """
        self.table_name = table_name or os.environ.get("ARTIFACTS_TABLE", "artifacts")
        self._dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self.table = self._dynamodb.Table(self.table_name)
    
    def put_artifact(
        self,
        artifact: ArtifactDB,
        overwrite: bool = True,
    ) -> ArtifactDB:
        """
        Create or update an artifact in DynamoDB.
        
        Args:
            artifact: The ArtifactDB model to store
            overwrite: If False, raises error if artifact exists
            
        Returns:
            The stored ArtifactDB model
            
        Raises:
            ArtifactAlreadyExistsError: If overwrite=False and artifact exists
            ArtifactRepositoryError: On DynamoDB errors
        """
        try:
            item = to_dynamodb_item(artifact)
            
            # Update the updated_at timestamp
            item["updated_at"] = datetime.utcnow().isoformat()
            
            if overwrite:
                self.table.put_item(Item=item)
            else:
                # Use condition to prevent overwrite
                self.table.put_item(
                    Item=item,
                    ConditionExpression="attribute_not_exists(artifact_id)"
                )
            
            logger.info(f"Stored artifact: {artifact.artifact_id}")
            return artifact
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ArtifactAlreadyExistsError(
                    f"Artifact already exists: {artifact.artifact_id}"
                )
            logger.error(f"Failed to put artifact: {e}")
            raise ArtifactRepositoryError(f"Failed to store artifact: {e}")
    
    def get_artifact(self, artifact_id: str) -> ArtifactDB:
        """
        Retrieve an artifact by its ID.
        
        Args:
            artifact_id: The artifact's unique identifier
            
        Returns:
            The ArtifactDB model
            
        Raises:
            ArtifactNotFoundError: If artifact doesn't exist
            ArtifactRepositoryError: On DynamoDB errors
        """
        try:
            response = self.table.get_item(Key={"artifact_id": artifact_id})
            item = response.get("Item")
            
            if not item:
                raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}")
            
            return from_dynamodb_item(item)
            
        except ArtifactNotFoundError:
            raise
        except ClientError as e:
            logger.error(f"Failed to get artifact {artifact_id}: {e}")
            raise ArtifactRepositoryError(f"Failed to retrieve artifact: {e}")
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact by its ID.
        
        Args:
            artifact_id: The artifact's unique identifier
            
        Returns:
            True if deleted successfully
            
        Raises:
            ArtifactNotFoundError: If artifact doesn't exist
            ArtifactRepositoryError: On DynamoDB errors
        """
        try:
            # Check if exists first
            self.get_artifact(artifact_id)
            
            self.table.delete_item(Key={"artifact_id": artifact_id})
            logger.info(f"Deleted artifact: {artifact_id}")
            return True
            
        except ArtifactNotFoundError:
            raise
        except ClientError as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            raise ArtifactRepositoryError(f"Failed to delete artifact: {e}")
    
    def query_by_name(
        self,
        name: str,
        limit: int = 100,
    ) -> list[ArtifactDB]:
        """
        Query artifacts by exact name using GSI_Name.
        
        Args:
            name: The exact artifact name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching ArtifactDB models
        """
        try:
            response = self.table.query(
                IndexName=self.GSI_NAME,
                KeyConditionExpression="name = :name",
                ExpressionAttributeValues={":name": name},
                Limit=limit,
            )
            
            items = response.get("Items", [])
            return [from_dynamodb_item(item) for item in items]
            
        except ClientError as e:
            logger.error(f"Failed to query by name {name}: {e}")
            raise ArtifactRepositoryError(f"Failed to query by name: {e}")
    
    def query_by_type(
        self,
        artifact_type: ArtifactType,
        limit: int = 100,
    ) -> list[ArtifactDB]:
        """
        Query artifacts by type using GSI_Type.
        
        Args:
            artifact_type: The artifact type to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching ArtifactDB models
        """
        try:
            type_value = artifact_type.value if isinstance(artifact_type, ArtifactType) else str(artifact_type)
            
            response = self.table.query(
                IndexName=self.GSI_TYPE,
                KeyConditionExpression="#type = :type",
                ExpressionAttributeNames={"#type": "type"},
                ExpressionAttributeValues={":type": type_value},
                Limit=limit,
            )
            
            items = response.get("Items", [])
            return [from_dynamodb_item(item) for item in items]
            
        except ClientError as e:
            logger.error(f"Failed to query by type {artifact_type}: {e}")
            raise ArtifactRepositoryError(f"Failed to query by type: {e}")
    
    def query_by_name_version(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> list[ArtifactDB]:
        """
        Query artifacts by name and optionally version using GSI_Name_Version.
        
        Args:
            name: The artifact name
            version: Optional exact version to match
            
        Returns:
            List of matching ArtifactDB models
        """
        try:
            if version:
                # Query with both name and version
                response = self.table.query(
                    IndexName=self.GSI_NAME_VERSION,
                    KeyConditionExpression="name = :name AND version = :version",
                    ExpressionAttributeValues={
                        ":name": name,
                        ":version": version,
                    },
                )
            else:
                # Query by name only, get all versions
                response = self.table.query(
                    IndexName=self.GSI_NAME_VERSION,
                    KeyConditionExpression="name = :name",
                    ExpressionAttributeValues={":name": name},
                )
            
            items = response.get("Items", [])
            return [from_dynamodb_item(item) for item in items]
            
        except ClientError as e:
            logger.error(f"Failed to query by name/version {name}/{version}: {e}")
            raise ArtifactRepositoryError(f"Failed to query by name/version: {e}")
    
    def scan_all(
        self,
        limit: Optional[int] = None,
    ) -> list[ArtifactDB]:
        """
        Scan all artifacts in the table.
        
        Note: This can be expensive for large tables. Use with caution.
        
        Args:
            limit: Optional maximum number of results
            
        Returns:
            List of all ArtifactDB models
        """
        try:
            artifacts = []
            scan_kwargs = {}
            
            if limit:
                scan_kwargs["Limit"] = limit
            
            # Handle pagination
            while True:
                response = self.table.scan(**scan_kwargs)
                items = response.get("Items", [])
                artifacts.extend([from_dynamodb_item(item) for item in items])
                
                # Check if we've hit the limit
                if limit and len(artifacts) >= limit:
                    artifacts = artifacts[:limit]
                    break
                
                # Check for more pages
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                scan_kwargs["ExclusiveStartKey"] = last_key
            
            return artifacts
            
        except ClientError as e:
            logger.error(f"Failed to scan artifacts: {e}")
            raise ArtifactRepositoryError(f"Failed to scan artifacts: {e}")
    
    def scan_with_regex_filter(
        self,
        name_pattern: str,
        limit: int = 100,
    ) -> list[ArtifactDB]:
        """
        Scan artifacts and filter by regex pattern on name.
        
        Note: This performs a full table scan and applies regex
        filtering client-side. Use GSI queries when possible.
        
        Args:
            name_pattern: Regex pattern to match against artifact names
            limit: Maximum number of results to return
            
        Returns:
            List of matching ArtifactDB models
            
        Raises:
            ValueError: If regex pattern is invalid
        """
        try:
            # Validate regex pattern
            try:
                pattern = re.compile(name_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
            
            matching = []
            scan_kwargs = {}
            
            while len(matching) < limit:
                response = self.table.scan(**scan_kwargs)
                items = response.get("Items", [])
                
                for item in items:
                    artifact = from_dynamodb_item(item)
                    if pattern.search(artifact.name):
                        matching.append(artifact)
                        if len(matching) >= limit:
                            break
                
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                scan_kwargs["ExclusiveStartKey"] = last_key
            
            return matching
            
        except ValueError:
            raise
        except ClientError as e:
            logger.error(f"Failed to scan with regex: {e}")
            raise ArtifactRepositoryError(f"Failed to scan with regex: {e}")
    
    def update_artifact_field(
        self,
        artifact_id: str,
        field_name: str,
        field_value: Any,
    ) -> ArtifactDB:
        """
        Update a single field on an artifact.
        
        Args:
            artifact_id: The artifact's unique identifier
            field_name: Name of the field to update
            field_value: New value for the field
            
        Returns:
            Updated ArtifactDB model
        """
        try:
            # Convert floats to Decimal for DynamoDB
            from src.models.artifact import _convert_floats_to_decimal
            converted_value = _convert_floats_to_decimal(field_value)
            
            response = self.table.update_item(
                Key={"artifact_id": artifact_id},
                UpdateExpression=f"SET #field = :value, updated_at = :now",
                ExpressionAttributeNames={"#field": field_name},
                ExpressionAttributeValues={
                    ":value": converted_value,
                    ":now": datetime.utcnow().isoformat(),
                },
                ReturnValues="ALL_NEW",
            )
            
            return from_dynamodb_item(response["Attributes"])
            
        except ClientError as e:
            logger.error(f"Failed to update artifact field: {e}")
            raise ArtifactRepositoryError(f"Failed to update artifact: {e}")
    
    def update_lineage(
        self,
        artifact_id: str,
        parents: Optional[list[str]] = None,
        children: Optional[list[str]] = None,
    ) -> ArtifactDB:
        """
        Update the lineage information for an artifact.
        
        Args:
            artifact_id: The artifact's unique identifier
            parents: New list of parent artifact IDs (or None to keep existing)
            children: New list of child artifact IDs (or None to keep existing)
            
        Returns:
            Updated ArtifactDB model
        """
        artifact = self.get_artifact(artifact_id)
        lineage = artifact.lineage or {"parents": [], "children": []}
        
        if parents is not None:
            lineage["parents"] = parents
        if children is not None:
            lineage["children"] = children
        
        return self.update_artifact_field(artifact_id, "lineage", lineage)
    
    def add_child_to_parent(
        self,
        parent_id: str,
        child_id: str,
    ) -> ArtifactDB:
        """
        Add a child reference to a parent artifact.
        
        Args:
            parent_id: The parent artifact's ID
            child_id: The child artifact's ID to add
            
        Returns:
            Updated parent ArtifactDB model
        """
        parent = self.get_artifact(parent_id)
        lineage = parent.lineage or {"parents": [], "children": []}
        
        if child_id not in lineage.get("children", []):
            lineage.setdefault("children", []).append(child_id)
            return self.update_artifact_field(parent_id, "lineage", lineage)
        
        return parent
    
    def remove_child_from_parent(
        self,
        parent_id: str,
        child_id: str,
    ) -> ArtifactDB:
        """
        Remove a child reference from a parent artifact.
        
        Args:
            parent_id: The parent artifact's ID
            child_id: The child artifact's ID to remove
            
        Returns:
            Updated parent ArtifactDB model
        """
        parent = self.get_artifact(parent_id)
        lineage = parent.lineage or {"parents": [], "children": []}
        
        if child_id in lineage.get("children", []):
            lineage["children"].remove(child_id)
            return self.update_artifact_field(parent_id, "lineage", lineage)
        
        return parent
    
    def increment_download_count(self, artifact_id: str) -> ArtifactDB:
        """
        Increment the download count for an artifact.
        
        Args:
            artifact_id: The artifact's unique identifier
            
        Returns:
            Updated ArtifactDB model
        """
        try:
            response = self.table.update_item(
                Key={"artifact_id": artifact_id},
                UpdateExpression=(
                    "SET download_info.download_count = if_not_exists(download_info.download_count, :zero) + :inc, "
                    "download_info.last_download_at = :now, "
                    "updated_at = :now"
                ),
                ExpressionAttributeValues={
                    ":zero": 0,
                    ":inc": 1,
                    ":now": datetime.utcnow().isoformat(),
                },
                ReturnValues="ALL_NEW",
            )
            
            return from_dynamodb_item(response["Attributes"])
            
        except ClientError as e:
            logger.error(f"Failed to increment download count: {e}")
            raise ArtifactRepositoryError(f"Failed to increment download count: {e}")


def create_artifacts_table(
    table_name: str = "artifacts",
    dynamodb_client: Optional[Any] = None,
) -> None:
    """
    Create the artifacts DynamoDB table with GSIs.
    
    This is a helper for local development and testing.
    Production tables should be created via CloudFormation/Terraform.
    
    Args:
        table_name: Name of the table to create
        dynamodb_client: Optional boto3 DynamoDB client
    """
    client = dynamodb_client or boto3.client("dynamodb")
    
    try:
        client.create_table(
            TableName=table_name,
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
                    "KeySchema": [
                        {"AttributeName": "name", "KeyType": "HASH"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "GSI_Type",
                    "KeySchema": [
                        {"AttributeName": "type", "KeyType": "HASH"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "GSI_Name_Version",
                    "KeySchema": [
                        {"AttributeName": "name", "KeyType": "HASH"},
                        {"AttributeName": "version", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )
        logger.info(f"Created table: {table_name}")
        
    except client.exceptions.ResourceInUseException:
        logger.info(f"Table already exists: {table_name}")
