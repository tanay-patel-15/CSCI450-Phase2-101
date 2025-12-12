"""
Unit tests for utility modules.

Tests versioning, cost computation, and lineage graph operations.
"""

import pytest

from src.utils.versioning import (
    matches_version_constraint,
    parse_version_constraint,
    compare_versions,
    get_latest_version,
    filter_versions_by_constraint,
    VersionConstraintError,
)
from src.utils.cost import (
    compute_cost,
    compute_cost_from_artifact,
    estimate_total_lineage_cost,
)
from src.utils.lineage_graph import (
    validate_no_cycles,
    compute_lineage_graph,
    get_root_artifacts,
    get_leaf_artifacts,
    topological_sort,
    CycleDetectedError,
)


class TestVersionMatching:
    """Tests for version constraint matching."""
    
    def test_exact_match(self):
        """Test exact version matching."""
        assert matches_version_constraint("1.2.3", "1.2.3") is True
        assert matches_version_constraint("1.2.3", "1.2.4") is False
    
    def test_greater_than_or_equal(self):
        """Test >= constraint."""
        assert matches_version_constraint("1.2.3", ">=1.0.0") is True
        assert matches_version_constraint("1.2.3", ">=1.2.3") is True
        assert matches_version_constraint("1.2.3", ">=2.0.0") is False
    
    def test_less_than(self):
        """Test < constraint."""
        assert matches_version_constraint("1.2.3", "<2.0.0") is True
        assert matches_version_constraint("2.0.0", "<2.0.0") is False
        assert matches_version_constraint("1.2.3", "<1.0.0") is False
    
    def test_bounded_range(self):
        """Test bounded range (two constraints)."""
        assert matches_version_constraint("1.5.0", ">=1.0.0 <2.0.0") is True
        assert matches_version_constraint("2.0.0", ">=1.0.0 <2.0.0") is False
        assert matches_version_constraint("0.9.0", ">=1.0.0 <2.0.0") is False
    
    def test_tilde_constraint(self):
        """Test tilde (~) constraint - patch level only."""
        # ~1.2.3 means >=1.2.3 <1.3.0
        assert matches_version_constraint("1.2.3", "~1.2.3") is True
        assert matches_version_constraint("1.2.9", "~1.2.3") is True
        assert matches_version_constraint("1.3.0", "~1.2.3") is False
        assert matches_version_constraint("1.2.2", "~1.2.3") is False
    
    def test_caret_constraint(self):
        """Test caret (^) constraint - minor level."""
        # ^1.2.3 means >=1.2.3 <2.0.0
        assert matches_version_constraint("1.2.3", "^1.2.3") is True
        assert matches_version_constraint("1.9.9", "^1.2.3") is True
        assert matches_version_constraint("2.0.0", "^1.2.3") is False
    
    def test_caret_zero_major(self):
        """Test caret with 0.x versions."""
        # ^0.2.3 means >=0.2.3 <0.3.0
        assert matches_version_constraint("0.2.5", "^0.2.3") is True
        assert matches_version_constraint("0.3.0", "^0.2.3") is False
    
    def test_invalid_version(self):
        """Test that invalid version returns False."""
        assert matches_version_constraint("invalid", ">=1.0.0") is False


class TestParseVersionConstraint:
    """Tests for version constraint parsing."""
    
    def test_parse_exact(self):
        """Test parsing exact version."""
        op, ver = parse_version_constraint("1.2.3")
        assert op == "=="
        assert ver == "1.2.3"
    
    def test_parse_comparison(self):
        """Test parsing comparison operators."""
        assert parse_version_constraint(">=1.0.0") == (">=", "1.0.0")
        assert parse_version_constraint("<2.0.0") == ("<", "2.0.0")
        assert parse_version_constraint("!=1.5.0") == ("!=", "1.5.0")
    
    def test_parse_tilde(self):
        """Test parsing tilde constraint."""
        assert parse_version_constraint("~1.2.3") == ("~", "1.2.3")
    
    def test_parse_caret(self):
        """Test parsing caret constraint."""
        assert parse_version_constraint("^1.2.3") == ("^", "1.2.3")


class TestCompareVersions:
    """Tests for version comparison."""
    
    def test_compare_less_than(self):
        """Test version less than comparison."""
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("1.2.3", "1.2.4") == -1
    
    def test_compare_equal(self):
        """Test version equality."""
        assert compare_versions("1.2.3", "1.2.3") == 0
    
    def test_compare_greater_than(self):
        """Test version greater than comparison."""
        assert compare_versions("2.0.0", "1.0.0") == 1


class TestGetLatestVersion:
    """Tests for getting latest version."""
    
    def test_get_latest(self):
        """Test getting latest version from list."""
        versions = ["1.0.0", "2.0.0", "1.5.0", "2.1.0"]
        assert get_latest_version(versions) == "2.1.0"
    
    def test_empty_list(self):
        """Test empty list returns None."""
        assert get_latest_version([]) is None
    
    def test_invalid_versions_skipped(self):
        """Test invalid versions are skipped."""
        versions = ["1.0.0", "invalid", "2.0.0"]
        assert get_latest_version(versions) == "2.0.0"


class TestFilterVersionsByConstraint:
    """Tests for filtering versions."""
    
    def test_filter_by_constraint(self):
        """Test filtering versions by constraint."""
        versions = ["1.0.0", "1.5.0", "2.0.0", "2.5.0"]
        filtered = filter_versions_by_constraint(versions, ">=1.5.0 <2.5.0")
        assert filtered == ["1.5.0", "2.0.0"]


class TestComputeCost:
    """Tests for cost computation."""
    
    def test_basic_cost(self):
        """Test basic cost calculation."""
        # 1MB = 0.001 cost units (with default multiplier)
        cost = compute_cost(1000000)
        assert cost == 0.001
    
    def test_cost_with_parents(self):
        """Test cost with parent costs."""
        cost = compute_cost(1000000, [0.5, 0.3])
        assert cost == 0.801
    
    def test_zero_size(self):
        """Test zero size artifact."""
        cost = compute_cost(0)
        assert cost == 0.0
    
    def test_custom_cost_per_byte(self):
        """Test custom cost multiplier."""
        cost = compute_cost(1000, cost_per_byte=0.001)
        assert cost == 1.0


class TestEstimateTotalLineageCost:
    """Tests for total lineage cost estimation."""
    
    def test_simple_lineage(self):
        """Test simple parent-child cost calculation."""
        costs = {"a": 1.0, "b": 2.0, "c": 3.0}
        graph = {"c": ["b"], "b": ["a"], "a": []}
        
        total = estimate_total_lineage_cost(costs, graph, "c")
        assert total == 6.0  # a + b + c
    
    def test_shared_ancestor(self):
        """Test that shared ancestors are not double-counted."""
        # Diamond dependency: d depends on b and c, both depend on a
        costs = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        graph = {"d": ["b", "c"], "b": ["a"], "c": ["a"], "a": []}
        
        total = estimate_total_lineage_cost(costs, graph, "d")
        assert total == 4.0  # a + b + c + d (a counted once)


class TestValidateNoCycles:
    """Tests for cycle detection."""
    
    def test_no_cycle(self):
        """Test valid DAG has no cycle."""
        # Simple chain: c -> b -> a
        def get_parents(artifact_id):
            parents = {"c": ["b"], "b": ["a"], "a": []}
            return parents.get(artifact_id, [])
        
        result = validate_no_cycles(["b"], "c", get_parents)
        assert result is True
    
    def test_direct_cycle(self):
        """Test direct cycle is detected."""
        def get_parents(artifact_id):
            # a -> b -> a (cycle)
            parents = {"a": ["b"], "b": ["a"]}
            return parents.get(artifact_id, [])
        
        with pytest.raises(CycleDetectedError):
            validate_no_cycles(["b"], "a", get_parents)
    
    def test_indirect_cycle(self):
        """Test indirect cycle is detected."""
        def get_parents(artifact_id):
            # a -> b -> c -> a (cycle)
            parents = {"a": ["b"], "b": ["c"], "c": ["a"]}
            return parents.get(artifact_id, [])
        
        with pytest.raises(CycleDetectedError):
            validate_no_cycles(["b"], "a", get_parents)


class TestComputeLineageGraph:
    """Tests for lineage graph computation."""
    
    def test_compute_graph(self):
        """Test computing full lineage graph."""
        # Create mock artifacts
        class MockArtifact:
            def __init__(self, aid, parents, children):
                self.artifact_id = aid
                self.lineage = {"parents": parents, "children": children}
        
        artifacts = {
            "a": MockArtifact("a", [], ["b"]),
            "b": MockArtifact("b", ["a"], ["c"]),
            "c": MockArtifact("c", ["b"], []),
        }
        
        def get_artifact(aid):
            return artifacts[aid]
        
        graph = compute_lineage_graph("b", get_artifact)
        
        assert "a" in graph["ancestors"]
        assert "c" in graph["descendants"]


class TestRootAndLeafDetection:
    """Tests for root and leaf artifact detection."""
    
    def test_get_roots(self):
        """Test finding root artifacts."""
        class MockArtifact:
            def __init__(self, aid, parents):
                self.artifact_id = aid
                self.lineage = {"parents": parents, "children": []}
        
        artifacts = [
            MockArtifact("a", []),  # root
            MockArtifact("b", ["a"]),
            MockArtifact("c", []),  # root
        ]
        
        roots = get_root_artifacts(artifacts)
        
        assert len(roots) == 2
        assert any(r.artifact_id == "a" for r in roots)
        assert any(r.artifact_id == "c" for r in roots)
    
    def test_get_leaves(self):
        """Test finding leaf artifacts."""
        class MockArtifact:
            def __init__(self, aid, children):
                self.artifact_id = aid
                self.lineage = {"parents": [], "children": children}
        
        artifacts = [
            MockArtifact("a", ["b"]),
            MockArtifact("b", []),  # leaf
            MockArtifact("c", []),  # leaf
        ]
        
        leaves = get_leaf_artifacts(artifacts)
        
        assert len(leaves) == 2


class TestTopologicalSort:
    """Tests for topological sorting."""
    
    def test_simple_sort(self):
        """Test simple topological sort."""
        class MockArtifact:
            def __init__(self, aid, parents):
                self.artifact_id = aid
                self.lineage = {"parents": parents}
        
        artifacts = [
            MockArtifact("c", ["b"]),
            MockArtifact("b", ["a"]),
            MockArtifact("a", []),
        ]
        
        order = topological_sort(artifacts)
        
        # a should come before b, b before c
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
    
    def test_sort_with_cycle(self):
        """Test that cycle raises error."""
        class MockArtifact:
            def __init__(self, aid, parents):
                self.artifact_id = aid
                self.lineage = {"parents": parents}
        
        artifacts = [
            MockArtifact("a", ["b"]),
            MockArtifact("b", ["a"]),
        ]
        
        with pytest.raises(CycleDetectedError):
            topological_sort(artifacts)
