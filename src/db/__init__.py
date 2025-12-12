"""
Database package for Artifact Data Model.

Exports the DynamoDB repository and related utilities.
"""

from db.artifact_repository import ArtifactRepository

__all__ = ["ArtifactRepository"]
