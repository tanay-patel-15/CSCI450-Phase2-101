"""
Semantic versioning utilities.

This module provides version constraint matching support for
artifact versioning, including exact matches, bounded ranges,
tilde, and caret constraints.

Uses the `packaging` library for robust version parsing and comparison.
"""

import re
from typing import Optional, Tuple

from packaging.version import Version, InvalidVersion


class VersionConstraintError(Exception):
    """Raised when a version constraint is invalid."""
    pass


def parse_version(version_str: str) -> Version:
    """
    Parse a version string into a Version object.
    
    Args:
        version_str: Semantic version string (e.g., "1.2.3")
        
    Returns:
        Parsed Version object
        
    Raises:
        VersionConstraintError: If version string is invalid
    """
    try:
        return Version(version_str)
    except InvalidVersion as e:
        raise VersionConstraintError(f"Invalid version: {version_str}") from e


def parse_version_constraint(constraint: str) -> Tuple[str, str]:
    """
    Parse a version constraint into operator and version components.
    
    Supported formats:
    - Exact: "1.2.3" -> ("==", "1.2.3")
    - Comparison: ">=1.2.3", "<2.0.0", etc.
    - Tilde: "~1.2.3" -> allows patch updates (~1.2.3 matches >=1.2.3 <1.3.0)
    - Caret: "^1.2.3" -> allows minor updates (^1.2.3 matches >=1.2.3 <2.0.0)
    
    Args:
        constraint: Version constraint string
        
    Returns:
        Tuple of (operator, version_string)
        
    Raises:
        VersionConstraintError: If constraint format is invalid
    """
    constraint = constraint.strip()
    
    # Tilde constraint (~1.2.3)
    if constraint.startswith("~"):
        return ("~", constraint[1:].strip())
    
    # Caret constraint (^1.2.3)
    if constraint.startswith("^"):
        return ("^", constraint[1:].strip())
    
    # Comparison operators
    operators = [">=", "<=", "!=", "==", ">", "<"]
    for op in operators:
        if constraint.startswith(op):
            return (op, constraint[len(op):].strip())
    
    # No operator means exact match
    return ("==", constraint)


def _expand_tilde_constraint(version_str: str) -> Tuple[str, str]:
    """
    Expand a tilde constraint to a version range.
    
    ~1.2.3 means >=1.2.3 and <1.3.0 (patch-level changes allowed)
    ~1.2 means >=1.2.0 and <1.3.0
    ~1 means >=1.0.0 and <2.0.0
    
    Args:
        version_str: Version string without the tilde
        
    Returns:
        Tuple of (min_version, max_version) strings
    """
    parts = version_str.split(".")
    
    if len(parts) >= 2:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0
        
        min_ver = f"{major}.{minor}.{patch}"
        max_ver = f"{major}.{minor + 1}.0"
    else:
        major = int(parts[0])
        min_ver = f"{major}.0.0"
        max_ver = f"{major + 1}.0.0"
    
    return (min_ver, max_ver)


def _expand_caret_constraint(version_str: str) -> Tuple[str, str]:
    """
    Expand a caret constraint to a version range.
    
    ^1.2.3 means >=1.2.3 and <2.0.0 (minor-level changes allowed)
    ^0.2.3 means >=0.2.3 and <0.3.0 (for 0.x versions, minor is breaking)
    ^0.0.3 means >=0.0.3 and <0.0.4 (for 0.0.x, patch is breaking)
    
    Args:
        version_str: Version string without the caret
        
    Returns:
        Tuple of (min_version, max_version) strings
    """
    parts = version_str.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    
    min_ver = f"{major}.{minor}.{patch}"
    
    if major != 0:
        # ^1.2.3 -> >=1.2.3, <2.0.0
        max_ver = f"{major + 1}.0.0"
    elif minor != 0:
        # ^0.2.3 -> >=0.2.3, <0.3.0
        max_ver = f"0.{minor + 1}.0"
    else:
        # ^0.0.3 -> >=0.0.3, <0.0.4
        max_ver = f"0.0.{patch + 1}"
    
    return (min_ver, max_ver)


def matches_version_constraint(version: str, constraint: str) -> bool:
    """
    Check if a version matches a constraint.
    
    Supports:
    - Exact match: "1.2.3"
    - Comparison: ">=1.0.0", "<2.0.0", etc.
    - Bounded ranges: ">=1.0.0 <2.0.0" (space-separated)
    - Tilde: "~1.2.3" (allows patch updates)
    - Caret: "^1.2.3" (allows minor updates for 1.x+)
    
    Args:
        version: The version to check (e.g., "1.2.3")
        constraint: The constraint to match against
        
    Returns:
        True if the version matches the constraint
        
    Raises:
        VersionConstraintError: If version or constraint is invalid
        
    Examples:
        >>> matches_version_constraint("1.2.3", "1.2.3")
        True
        >>> matches_version_constraint("1.2.3", ">=1.0.0")
        True
        >>> matches_version_constraint("1.2.3", ">=1.0.0 <2.0.0")
        True
        >>> matches_version_constraint("1.2.3", "~1.2.0")
        True
        >>> matches_version_constraint("1.3.0", "~1.2.0")
        False
        >>> matches_version_constraint("1.5.0", "^1.2.3")
        True
        >>> matches_version_constraint("2.0.0", "^1.2.3")
        False
    """
    try:
        ver = parse_version(version)
    except VersionConstraintError:
        return False
    
    # Handle bounded ranges (space-separated constraints)
    if " " in constraint:
        parts = constraint.split()
        return all(matches_version_constraint(version, part) for part in parts)
    
    op, constraint_ver = parse_version_constraint(constraint)
    
    try:
        if op == "~":
            min_ver, max_ver = _expand_tilde_constraint(constraint_ver)
            return Version(min_ver) <= ver < Version(max_ver)
        
        elif op == "^":
            min_ver, max_ver = _expand_caret_constraint(constraint_ver)
            return Version(min_ver) <= ver < Version(max_ver)
        
        elif op == "==":
            return ver == Version(constraint_ver)
        
        elif op == "!=":
            return ver != Version(constraint_ver)
        
        elif op == ">=":
            return ver >= Version(constraint_ver)
        
        elif op == "<=":
            return ver <= Version(constraint_ver)
        
        elif op == ">":
            return ver > Version(constraint_ver)
        
        elif op == "<":
            return ver < Version(constraint_ver)
        
        else:
            raise VersionConstraintError(f"Unknown operator: {op}")
            
    except InvalidVersion as e:
        raise VersionConstraintError(f"Invalid constraint version: {constraint_ver}") from e


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    
    Args:
        version1: First version string
        version2: Second version string
        
    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
         
    Raises:
        VersionConstraintError: If either version is invalid
    """
    v1 = parse_version(version1)
    v2 = parse_version(version2)
    
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def get_latest_version(versions: list[str]) -> Optional[str]:
    """
    Get the latest (highest) version from a list.
    
    Args:
        versions: List of version strings
        
    Returns:
        The highest version string, or None if list is empty
    """
    if not versions:
        return None
    
    valid_versions = []
    for v in versions:
        try:
            valid_versions.append((parse_version(v), v))
        except VersionConstraintError:
            continue
    
    if not valid_versions:
        return None
    
    return max(valid_versions, key=lambda x: x[0])[1]


def filter_versions_by_constraint(
    versions: list[str],
    constraint: str,
) -> list[str]:
    """
    Filter a list of versions by a constraint.
    
    Args:
        versions: List of version strings
        constraint: Version constraint to filter by
        
    Returns:
        List of versions that match the constraint
    """
    return [v for v in versions if matches_version_constraint(v, constraint)]
