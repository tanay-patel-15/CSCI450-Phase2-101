"""
S3 utilities for artifact storage and downloads.

This module provides helpers for generating pre-signed URLs,
tracking downloads, and managing S3 storage for artifacts.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_BUCKET = os.environ.get("BUCKET_NAME", "project-models-group102")
DEFAULT_EXPIRATION = 3600  # 1 hour


def get_s3_client() -> Any:
    """Get a boto3 S3 client."""
    return boto3.client("s3")


def generate_presigned_url(
    s3_key: str,
    bucket_name: Optional[str] = None,
    expiration: int = DEFAULT_EXPIRATION,
    http_method: str = "GET",
) -> str:
    """
    Generate a pre-signed URL for an S3 object.
    
    Args:
        s3_key: The S3 object key
        bucket_name: S3 bucket name (defaults to BUCKET_NAME env var)
        expiration: URL expiration time in seconds (default: 3600)
        http_method: HTTP method for the URL (default: GET)
        
    Returns:
        Pre-signed URL string
        
    Raises:
        ValueError: If s3_key is empty
        ClientError: On S3 errors
    """
    if not s3_key:
        raise ValueError("s3_key cannot be empty")
    
    bucket = bucket_name or DEFAULT_BUCKET
    s3_client = get_s3_client()
    
    try:
        if http_method == "GET":
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
        elif http_method == "PUT":
            url = s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {http_method}")
        
        logger.debug(f"Generated presigned URL for {s3_key}")
        return url
        
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
        raise


def get_object_size(
    s3_key: str,
    bucket_name: Optional[str] = None,
) -> int:
    """
    Get the size of an S3 object in bytes.
    
    Args:
        s3_key: The S3 object key
        bucket_name: S3 bucket name (defaults to BUCKET_NAME env var)
        
    Returns:
        Object size in bytes
        
    Raises:
        ClientError: If object doesn't exist or on S3 errors
    """
    bucket = bucket_name or DEFAULT_BUCKET
    s3_client = get_s3_client()
    
    try:
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        return response.get("ContentLength", 0)
    except ClientError as e:
        logger.error(f"Failed to get object size for {s3_key}: {e}")
        raise


def object_exists(
    s3_key: str,
    bucket_name: Optional[str] = None,
) -> bool:
    """
    Check if an S3 object exists.
    
    Args:
        s3_key: The S3 object key
        bucket_name: S3 bucket name (defaults to BUCKET_NAME env var)
        
    Returns:
        True if object exists, False otherwise
    """
    bucket = bucket_name or DEFAULT_BUCKET
    s3_client = get_s3_client()
    
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        logger.error(f"Error checking object existence for {s3_key}: {e}")
        raise


def build_artifact_s3_key(
    artifact_id: str,
    artifact_type: str,
    version: str,
    filename: Optional[str] = None,
) -> str:
    """
    Build an S3 key for an artifact.
    
    Creates a hierarchical key structure:
        artifacts/{type}/{artifact_id}/{version}/{filename}
    
    Args:
        artifact_id: The artifact's unique identifier
        artifact_type: Type of artifact (MODEL, DATASET, CODE)
        version: Artifact version
        filename: Optional filename (defaults to "artifact.bin")
        
    Returns:
        S3 key string
    """
    type_lower = artifact_type.lower()
    file = filename or "artifact.bin"
    return f"artifacts/{type_lower}/{artifact_id}/{version}/{file}"


def increment_download_count(
    artifact_id: str,
    repository: Any,
) -> int:
    """
    Increment the download count for an artifact.
    
    Args:
        artifact_id: The artifact's unique identifier
        repository: ArtifactRepository instance
        
    Returns:
        New download count
    """
    artifact = repository.increment_download_count(artifact_id)
    download_info = artifact.download_info or {}
    return download_info.get("download_count", 0)


def update_download_info(
    artifact_id: str,
    repository: Any,
    s3_key: Optional[str] = None,
    download_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Update download information for an artifact.
    
    Args:
        artifact_id: The artifact's unique identifier
        repository: ArtifactRepository instance
        s3_key: Optional new S3 key
        download_url: Optional new download URL
        
    Returns:
        Updated download_info dictionary
    """
    artifact = repository.get_artifact(artifact_id)
    download_info = artifact.download_info or {}
    
    if s3_key is not None:
        download_info["s3_key"] = s3_key
    if download_url is not None:
        download_info["download_url"] = download_url
    
    updated = repository.update_artifact_field(
        artifact_id, 
        "download_info", 
        download_info
    )
    return updated.download_info


def prepare_download_response(
    artifact_id: str,
    repository: Any,
    bucket_name: Optional[str] = None,
    expiration: int = DEFAULT_EXPIRATION,
) -> dict[str, Any]:
    """
    Prepare a download response for an artifact.
    
    This function:
    1. Retrieves the artifact
    2. Generates a presigned URL
    3. Increments the download count
    4. Updates the last download time
    
    Args:
        artifact_id: The artifact's unique identifier
        repository: ArtifactRepository instance
        bucket_name: Optional S3 bucket name
        expiration: URL expiration in seconds
        
    Returns:
        Dictionary with download URL and metadata
    """
    # Get artifact
    artifact = repository.get_artifact(artifact_id)
    download_info = artifact.download_info or {}
    s3_key = download_info.get("s3_key", "")
    
    if not s3_key:
        # Build default S3 key if not set
        s3_key = build_artifact_s3_key(
            artifact_id,
            artifact.type,
            artifact.version,
        )
    
    # Generate presigned URL
    download_url = generate_presigned_url(
        s3_key,
        bucket_name=bucket_name,
        expiration=expiration,
    )
    
    # Increment download count (also updates last_download_at)
    repository.increment_download_count(artifact_id)
    
    return {
        "artifact_id": artifact_id,
        "name": artifact.name,
        "version": artifact.version,
        "download_url": download_url,
        "expires_in": expiration,
        "size_bytes": artifact.metadata.get("size_bytes", 0) if artifact.metadata else 0,
    }
