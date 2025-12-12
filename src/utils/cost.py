"""
Cost computation utilities for artifacts.

This module provides functions to compute the cost of an artifact
based on its size and the costs of its parent artifacts.
"""

from typing import Any, Optional


def compute_cost(
    artifact_size_bytes: int,
    parent_costs: Optional[list[float]] = None,
    cost_per_byte: float = 1e-9,
) -> float:
    """
    Compute the cost of an artifact.
    
    The cost is calculated as:
        artifact_size (in cost units) + sum(parent_costs)
    
    Args:
        artifact_size_bytes: Size of the artifact in bytes
        parent_costs: List of costs from parent artifacts
        cost_per_byte: Cost multiplier per byte (default: 1e-9)
        
    Returns:
        Total computed cost as a float
        
    Examples:
        >>> compute_cost(1000000)  # 1MB artifact, no parents
        0.001
        >>> compute_cost(1000000, [0.5, 0.3])  # 1MB with two parents
        0.801
    """
    if parent_costs is None:
        parent_costs = []
    
    # Convert bytes to cost units
    size_cost = artifact_size_bytes * cost_per_byte
    
    # Sum parent costs
    parent_total = sum(parent_costs)
    
    return round(size_cost + parent_total, 6)


def compute_cost_from_artifact(
    artifact: Any,
    get_parent_cost_fn: Optional[callable] = None,
) -> float:
    """
    Compute cost from an artifact object.
    
    This is a convenience function that extracts the necessary
    fields from an artifact-like object.
    
    Args:
        artifact: An object with metadata.size_bytes and lineage.parents
        get_parent_cost_fn: Optional function to get cost of a parent by ID
        
    Returns:
        Computed cost
    """
    # Extract size from artifact
    size_bytes = 0
    if hasattr(artifact, "metadata"):
        metadata = artifact.metadata
        if isinstance(metadata, dict):
            size_bytes = metadata.get("size_bytes", 0)
        elif hasattr(metadata, "size_bytes"):
            size_bytes = metadata.size_bytes
    
    # Get parent costs if function provided
    parent_costs = []
    if get_parent_cost_fn is not None:
        parents = []
        if hasattr(artifact, "lineage"):
            lineage = artifact.lineage
            if isinstance(lineage, dict):
                parents = lineage.get("parents", [])
            elif hasattr(lineage, "parents"):
                parents = lineage.parents
        
        for parent_id in parents:
            try:
                parent_cost = get_parent_cost_fn(parent_id)
                if parent_cost is not None:
                    parent_costs.append(float(parent_cost))
            except Exception:
                # Skip parent if cost cannot be retrieved
                pass
    
    return compute_cost(size_bytes, parent_costs)


def estimate_total_lineage_cost(
    artifact_costs: dict[str, float],
    lineage_graph: dict[str, list[str]],
    target_artifact_id: str,
) -> float:
    """
    Estimate the total cost including all ancestors.
    
    Traverses the lineage graph to sum up all ancestor costs,
    avoiding double-counting shared ancestors.
    
    Args:
        artifact_costs: Mapping of artifact_id to cost
        lineage_graph: Mapping of artifact_id to list of parent IDs
        target_artifact_id: The artifact to calculate total cost for
        
    Returns:
        Total cost including all unique ancestors
    """
    visited = set()
    total_cost = 0.0
    
    def visit(artifact_id: str) -> None:
        nonlocal total_cost
        if artifact_id in visited:
            return
        visited.add(artifact_id)
        
        # Add this artifact's cost
        total_cost += artifact_costs.get(artifact_id, 0.0)
        
        # Visit parents
        for parent_id in lineage_graph.get(artifact_id, []):
            visit(parent_id)
    
    visit(target_artifact_id)
    return round(total_cost, 6)
