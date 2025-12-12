"""
Lineage graph utilities for artifact relationships.

This module provides functions to compute and validate the
directed acyclic graph (DAG) of artifact lineage relationships.
"""

from collections import deque
from typing import Any, Optional


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the lineage graph."""
    pass


def validate_no_cycles(
    parents: list[str],
    artifact_id: str,
    get_artifact_parents_fn: callable,
) -> bool:
    """
    Validate that adding parents won't create a cycle.
    
    Uses BFS to traverse ancestors and check if the artifact
    would appear in its own ancestry.
    
    Args:
        parents: List of proposed parent artifact IDs
        artifact_id: The artifact that would have these parents
        get_artifact_parents_fn: Function to get parents of an artifact
        
    Returns:
        True if no cycle would be created
        
    Raises:
        CycleDetectedError: If adding these parents would create a cycle
    """
    # Check if artifact_id is in any parent's ancestry
    visited = set()
    queue = deque(parents)
    
    while queue:
        current = queue.popleft()
        
        if current == artifact_id:
            raise CycleDetectedError(
                f"Adding parents would create a cycle: {artifact_id} found in ancestry"
            )
        
        if current in visited:
            continue
        visited.add(current)
        
        # Get parents of current and add to queue
        try:
            current_parents = get_artifact_parents_fn(current)
            if current_parents:
                queue.extend(current_parents)
        except Exception:
            # If we can't get parents, skip this node
            pass
    
    return True


def compute_lineage_graph(
    artifact_id: str,
    get_artifact_fn: callable,
    max_depth: int = 100,
) -> dict[str, Any]:
    """
    Compute the full lineage graph for an artifact.
    
    Returns both ancestors (upward traversal via parents) and
    descendants (downward traversal via children).
    
    Args:
        artifact_id: The artifact to compute lineage for
        get_artifact_fn: Function to get an artifact by ID
        max_depth: Maximum depth to traverse (prevents infinite loops)
        
    Returns:
        Dictionary with:
        - ancestors: List of all ancestor artifact IDs
        - descendants: List of all descendant artifact IDs
        - ancestor_tree: Nested dict showing parent relationships
        - descendant_tree: Nested dict showing child relationships
        - depth: Maximum depth of the lineage tree
    """
    ancestors = []
    descendants = []
    ancestor_tree = {}
    descendant_tree = {}
    max_seen_depth = 0
    
    # Traverse ancestors (via parents)
    def traverse_ancestors(aid: str, depth: int) -> dict:
        nonlocal max_seen_depth
        max_seen_depth = max(max_seen_depth, depth)
        
        if depth > max_depth:
            return {}
        
        try:
            artifact = get_artifact_fn(aid)
            lineage = artifact.lineage if hasattr(artifact, 'lineage') else {}
            if isinstance(lineage, dict):
                parents = lineage.get("parents", [])
            else:
                parents = lineage.parents if hasattr(lineage, 'parents') else []
            
            result = {}
            for parent_id in parents:
                if parent_id not in ancestors:
                    ancestors.append(parent_id)
                result[parent_id] = traverse_ancestors(parent_id, depth + 1)
            return result
        except Exception:
            return {}
    
    # Traverse descendants (via children)
    def traverse_descendants(aid: str, depth: int) -> dict:
        nonlocal max_seen_depth
        max_seen_depth = max(max_seen_depth, depth)
        
        if depth > max_depth:
            return {}
        
        try:
            artifact = get_artifact_fn(aid)
            lineage = artifact.lineage if hasattr(artifact, 'lineage') else {}
            if isinstance(lineage, dict):
                children = lineage.get("children", [])
            else:
                children = lineage.children if hasattr(lineage, 'children') else []
            
            result = {}
            for child_id in children:
                if child_id not in descendants:
                    descendants.append(child_id)
                result[child_id] = traverse_descendants(child_id, depth + 1)
            return result
        except Exception:
            return {}
    
    ancestor_tree = traverse_ancestors(artifact_id, 0)
    descendant_tree = traverse_descendants(artifact_id, 0)
    
    return {
        "artifact_id": artifact_id,
        "ancestors": ancestors,
        "descendants": descendants,
        "ancestor_tree": ancestor_tree,
        "descendant_tree": descendant_tree,
        "depth": max_seen_depth,
    }


def update_children_references(
    parent_id: str,
    child_id: str,
    repository: Any,
    remove: bool = False,
) -> None:
    """
    Update the children references on a parent artifact.
    
    When an artifact declares a parent, this function should be
    called to add the child reference to the parent. The reverse
    operation (remove) is used when a parent is removed.
    
    Args:
        parent_id: The parent artifact's ID
        child_id: The child artifact's ID
        repository: ArtifactRepository instance
        remove: If True, remove the child reference instead of adding
    """
    if remove:
        repository.remove_child_from_parent(parent_id, child_id)
    else:
        repository.add_child_to_parent(parent_id, child_id)


def sync_lineage_references(
    artifact_id: str,
    old_parents: list[str],
    new_parents: list[str],
    repository: Any,
) -> None:
    """
    Synchronize lineage references when parents change.
    
    Updates the children lists of affected parent artifacts
    when a child's parent list is modified.
    
    Args:
        artifact_id: The artifact whose parents are changing
        old_parents: Previous list of parent IDs
        new_parents: New list of parent IDs
        repository: ArtifactRepository instance
    """
    old_set = set(old_parents)
    new_set = set(new_parents)
    
    # Parents that were removed
    for removed_parent in old_set - new_set:
        update_children_references(removed_parent, artifact_id, repository, remove=True)
    
    # Parents that were added
    for added_parent in new_set - old_set:
        update_children_references(added_parent, artifact_id, repository, remove=False)


def get_root_artifacts(
    artifacts: list[Any],
) -> list[Any]:
    """
    Get all root artifacts (those with no parents).
    
    Args:
        artifacts: List of artifact objects with lineage info
        
    Returns:
        List of artifacts that have no parents
    """
    roots = []
    for artifact in artifacts:
        lineage = artifact.lineage if hasattr(artifact, 'lineage') else {}
        if isinstance(lineage, dict):
            parents = lineage.get("parents", [])
        else:
            parents = lineage.parents if hasattr(lineage, 'parents') else []
        
        if not parents:
            roots.append(artifact)
    
    return roots


def get_leaf_artifacts(
    artifacts: list[Any],
) -> list[Any]:
    """
    Get all leaf artifacts (those with no children).
    
    Args:
        artifacts: List of artifact objects with lineage info
        
    Returns:
        List of artifacts that have no children
    """
    leaves = []
    for artifact in artifacts:
        lineage = artifact.lineage if hasattr(artifact, 'lineage') else {}
        if isinstance(lineage, dict):
            children = lineage.get("children", [])
        else:
            children = lineage.children if hasattr(lineage, 'children') else []
        
        if not children:
            leaves.append(artifact)
    
    return leaves


def topological_sort(
    artifacts: list[Any],
    get_parents_fn: Optional[callable] = None,
) -> list[str]:
    """
    Perform topological sort on artifacts by lineage.
    
    Returns artifact IDs in dependency order (parents before children).
    
    Args:
        artifacts: List of artifact objects
        get_parents_fn: Optional function to get parents from an artifact
        
    Returns:
        List of artifact IDs in topological order
        
    Raises:
        CycleDetectedError: If a cycle is detected
    """
    # Build adjacency list and in-degree count
    in_degree = {}
    adj_list = {}
    artifact_ids = set()
    
    for artifact in artifacts:
        aid = artifact.artifact_id if hasattr(artifact, 'artifact_id') else str(artifact)
        artifact_ids.add(aid)
        
        if get_parents_fn:
            parents = get_parents_fn(artifact)
        else:
            lineage = artifact.lineage if hasattr(artifact, 'lineage') else {}
            if isinstance(lineage, dict):
                parents = lineage.get("parents", [])
            else:
                parents = lineage.parents if hasattr(lineage, 'parents') else []
        
        in_degree[aid] = len(parents)
        
        for parent in parents:
            if parent not in adj_list:
                adj_list[parent] = []
            adj_list[parent].append(aid)
    
    # Kahn's algorithm
    queue = deque([aid for aid in artifact_ids if in_degree.get(aid, 0) == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for child in adj_list.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    if len(result) != len(artifact_ids):
        raise CycleDetectedError("Cycle detected in artifact lineage")
    
    return result
