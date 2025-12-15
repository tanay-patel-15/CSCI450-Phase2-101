"""
Lineage tracking models for artifact relationships.

This module defines the lineage structure that supports a
directed acyclic graph (DAG) of artifact relationships,
tracking parent-child dependencies between artifacts.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class LineageInfo(BaseModel):
    """
    Lineage information for tracking artifact relationships.
    
    Maintains bidirectional references between artifacts:
    - parents: List of artifact IDs that this artifact depends on
    - children: List of artifact IDs that depend on this artifact
    
    The children list is maintained automatically by the system
    when artifacts declare parents during creation or update.
    """
    model_config = ConfigDict(extra="forbid")
    
    parents: list[str] = Field(
        default_factory=list,
        description="List of parent artifact IDs (dependencies)"
    )
    children: list[str] = Field(
        default_factory=list,
        description="List of child artifact IDs (dependents, auto-maintained)"
    )

    def add_parent(self, parent_id: str) -> bool:
        """
        Add a parent artifact ID to the lineage.
        
        Args:
            parent_id: The artifact_id of the parent to add
            
        Returns:
            True if the parent was added, False if already present
        """
        if parent_id not in self.parents:
            self.parents.append(parent_id)
            return True
        return False

    def remove_parent(self, parent_id: str) -> bool:
        """
        Remove a parent artifact ID from the lineage.
        
        Args:
            parent_id: The artifact_id of the parent to remove
            
        Returns:
            True if the parent was removed, False if not found
        """
        if parent_id in self.parents:
            self.parents.remove(parent_id)
            return True
        return False

    def add_child(self, child_id: str) -> bool:
        """
        Add a child artifact ID to the lineage.
        
        This is typically called by the system when another artifact
        declares this artifact as a parent.
        
        Args:
            child_id: The artifact_id of the child to add
            
        Returns:
            True if the child was added, False if already present
        """
        if child_id not in self.children:
            self.children.append(child_id)
            return True
        return False

    def remove_child(self, child_id: str) -> bool:
        """
        Remove a child artifact ID from the lineage.
        
        Args:
            child_id: The artifact_id of the child to remove
            
        Returns:
            True if the child was removed, False if not found
        """
        if child_id in self.children:
            self.children.remove(child_id)
            return True
        return False

    def has_parent(self, parent_id: str) -> bool:
        """Check if the given artifact is a parent."""
        return parent_id in self.parents

    def has_child(self, child_id: str) -> bool:
        """Check if the given artifact is a child."""
        return child_id in self.children

    @property
    def parent_count(self) -> int:
        """Get the number of parents."""
        return len(self.parents)

    @property
    def child_count(self) -> int:
        """Get the number of children."""
        return len(self.children)

    @property
    def is_root(self) -> bool:
        """Check if this artifact has no parents (is a root node)."""
        return len(self.parents) == 0

    @property
    def is_leaf(self) -> bool:
        """Check if this artifact has no children (is a leaf node)."""
        return len(self.children) == 0

    def to_dict(self) -> dict[str, list[str]]:
        """
        Convert to dictionary for DynamoDB storage.
        
        Returns:
            Dictionary with parents and children lists
        """
        return {
            "parents": list(self.parents),
            "children": list(self.children),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "LineageInfo":
        """
        Create from dictionary (e.g., from DynamoDB).
        
        Args:
            data: Dictionary with optional parents and children keys
            
        Returns:
            LineageInfo instance
        """
        if not data:
            return cls()
        
        return cls(
            parents=data.get("parents", []),
            children=data.get("children", []),
        )

    def merge(self, other: "LineageInfo") -> "LineageInfo":
        """
        Merge another LineageInfo into this one.
        
        Combines the parents and children from both, removing duplicates.
        
        Args:
            other: Another LineageInfo to merge
            
        Returns:
            New LineageInfo with combined relationships
        """
        merged_parents = list(set(self.parents + other.parents))
        merged_children = list(set(self.children + other.children))
        return LineageInfo(parents=merged_parents, children=merged_children)