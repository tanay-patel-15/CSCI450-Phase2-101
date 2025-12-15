"""
Metrics storage models for Phase 1 and Phase 2 metrics.

This module defines the nested metric structures stored within
each artifact, supporting both the original Phase 1 metrics
and the new Phase 2 metrics.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class SizeScore(BaseModel):
    """
    Size score by hardware target.
    
    Represents how well the artifact runs on different hardware
    configurations, scored from 0.0 to 1.0.
    """
    model_config = ConfigDict(extra="allow")
    
    raspberry_pi: float = Field(default=0.0, ge=0.0, le=1.0)
    jetson_nano: float = Field(default=0.0, ge=0.0, le=1.0)
    desktop_pc: float = Field(default=0.0, ge=0.0, le=1.0)
    aws_server: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for storage."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "SizeScore":
        """Create from dictionary."""
        return cls(**data)


class Phase1Metrics(BaseModel):
    """
    Phase 1 metrics from the original trustworthiness evaluation.
    
    These metrics are computed during the ingest pipeline and
    contribute to the overall net_score calculation.
    """
    model_config = ConfigDict(extra="forbid")
    
    ramp_up_time: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ease of getting started with the artifact (0-1)"
    )
    bus_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Risk score based on contributor diversity (0-1)"
    )
    license_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="License compatibility and clarity score (0-1)"
    )
    performance_claims: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Verification of stated performance claims (0-1)"
    )
    dataset_and_code_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Availability of training data and code (0-1)"
    )
    dataset_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality assessment of training data (0-1)"
    )
    code_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Code quality metrics (linting, tests, etc.) (0-1)"
    )
    size_score: dict[str, float] = Field(
        default_factory=lambda: {
            "raspberry_pi": 0.0,
            "jetson_nano": 0.0,
            "desktop_pc": 0.0,
            "aws_server": 0.0,
        },
        description="Size score by hardware target"
    )
    net_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted average of all Phase 1 metrics"
    )

    @field_validator("size_score", mode="before")
    @classmethod
    def ensure_size_score_dict(cls, v: Any) -> dict[str, float]:
        """Ensure size_score is a proper dictionary."""
        if v is None:
            return {
                "raspberry_pi": 0.0,
                "jetson_nano": 0.0,
                "desktop_pc": 0.0,
                "aws_server": 0.0,
            }
        if isinstance(v, SizeScore):
            return v.to_dict()
        return dict(v)


class Phase2Metrics(BaseModel):
    """
    Phase 2 new metrics for enhanced trustworthiness evaluation.
    
    These metrics extend the original Phase 1 metrics with
    additional trust signals specific to Phase 2 requirements.
    """
    model_config = ConfigDict(extra="forbid")
    
    reproducibility: float = Field(
        default=0.0,
        description="Reproducibility score: 0.0 (not reproducible), 0.5 (partially), 1.0 (fully)"
    )
    reviewedness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Code review coverage (PR-reviewed LOC / total LOC)"
    )
    treescale_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average of parent metrics (lineage-based trust)"
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Measured inference/execution latency in milliseconds"
    )

    @field_validator("reproducibility", mode="before")
    @classmethod
    def validate_reproducibility(cls, v: Any) -> float:
        """Validate reproducibility is one of the allowed values."""
        if v is None:
            return 0.0
        v = float(v)
        # Allow values close to 0.0, 0.5, or 1.0 (with floating point tolerance)
        allowed = [0.0, 0.5, 1.0]
        for allowed_val in allowed:
            if abs(v - allowed_val) < 0.001:
                return allowed_val
        # If not close to an allowed value, clamp to nearest
        if v < 0.25:
            return 0.0
        elif v < 0.75:
            return 0.5
        else:
            return 1.0


class ArtifactMetrics(BaseModel):
    """
    Combined metrics container for an artifact.
    
    Contains both Phase 1 and Phase 2 metrics in a nested structure
    suitable for DynamoDB storage.
    """
    model_config = ConfigDict(extra="forbid")
    
    phase1: Phase1Metrics = Field(
        default_factory=Phase1Metrics,
        description="Phase 1 trustworthiness metrics"
    )
    phase2: Phase2Metrics = Field(
        default_factory=Phase2Metrics,
        description="Phase 2 extended metrics"
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for DynamoDB storage.
        
        Returns:
            Nested dictionary with phase1 and phase2 metrics
        """
        return {
            "phase1": self.phase1.model_dump(),
            "phase2": self.phase2.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "ArtifactMetrics":
        """
        Create from dictionary (e.g., from DynamoDB).
        
        Args:
            data: Dictionary with optional phase1 and phase2 keys
            
        Returns:
            ArtifactMetrics instance
        """
        if not data:
            return cls()
        
        phase1_data = data.get("phase1", {})
        phase2_data = data.get("phase2", {})
        
        return cls(
            phase1=Phase1Metrics(**phase1_data) if phase1_data else Phase1Metrics(),
            phase2=Phase2Metrics(**phase2_data) if phase2_data else Phase2Metrics(),
        )

    def compute_net_score(
        self,
        weights: Optional[dict[str, float]] = None
    ) -> float:
        """
        Compute the weighted net score from Phase 1 metrics.
        
        Args:
            weights: Optional custom weights. If not provided, uses defaults.
            
        Returns:
            Weighted average score between 0.0 and 1.0
        """
        if weights is None:
            weights = {
                "ramp_up_time": 0.15,
                "bus_factor": 0.15,
                "license_score": 0.15,
                "performance_claims": 0.15,
                "dataset_and_code_score": 0.1,
                "dataset_quality": 0.1,
                "code_quality": 0.1,
                "size_score": 0.1,
            }
        
        # Compute average size score
        size_scores = self.phase1.size_score
        size_avg = sum(size_scores.values()) / max(len(size_scores), 1)
        
        score = 0.0
        score += weights.get("ramp_up_time", 0) * self.phase1.ramp_up_time
        score += weights.get("bus_factor", 0) * self.phase1.bus_factor
        score += weights.get("license_score", 0) * self.phase1.license_score
        score += weights.get("performance_claims", 0) * self.phase1.performance_claims
        score += weights.get("dataset_and_code_score", 0) * self.phase1.dataset_and_code_score
        score += weights.get("dataset_quality", 0) * self.phase1.dataset_quality
        score += weights.get("code_quality", 0) * self.phase1.code_quality
        score += weights.get("size_score", 0) * size_avg
        
        return round(score, 3)
    
    """
Metrics computation for artifacts.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# In src/metrics.py

def compute_metrics_for_model(url: str) -> dict:
    """
    Given a Hugging Face model URL, fetch metadata and compute metrics.
    """
    model_name = url.rstrip("/").split("/")[-1]
    start = time.time()

    # 1. Attempt to fetch real metadata from HF
    try:
        # Use the HF API to get real details
        api_url = f"{HF_API}{model_name}"
        response = requests.get(api_url, timeout=5) # Short timeout for speed
        if response.status_code == 200:
            data = response.json()
        else:
            data = {}
    except Exception:
        data = {}

    card_data = data.get("cardData", {})
    # Fallback to empty string if not found
    long_desc = str(card_data.get("long_description", "") or "")
    tags = data.get("tags", [])

    # --- METRIC 1: Ramp Up (Documentation Length) ---
    # If description is long (>5000 chars), score is high.
    ramp_up_time = min(len(long_desc) / 5000, 1.0)
    if ramp_up_time < 0.2: ramp_up_time = 0.2 # Floor

    # --- METRIC 2: Bus Factor (Files/Downloads) ---
    # Use 'siblings' (file count) or 'downloads' as proxy
    file_count = len(data.get("siblings", []))
    bus_factor = min(file_count / 10, 1.0)
    if bus_factor < 0.2: bus_factor = 0.2

    # --- METRIC 3: License ---
    # Explicit check
    license_score = 1.0 if data.get("license") else 0.0

    # --- METRIC 4: Performance Claims (Keyword Search) ---
    # Look for "accuracy", "F1", "speed" in the README
    perf_keywords = ["accuracy", "f1", "precision", "recall", "latency", "benchmark", "sota"]
    performance_claims = 0.8 if any(k in long_desc.lower() for k in perf_keywords) else 0.3

    # --- METRIC 5: Code/Dataset Quality (Tags) ---
    has_framework = any(f in tags for f in ["pytorch", "tensorflow", "jax", "onnx"])
    code_quality = 0.8 if has_framework else 0.4
    
    dataset_quality = 0.8 if "dataset" in tags else 0.4
    
    # Combined score
    dataset_and_code_score = (code_quality + dataset_quality) / 2

    # --- Size Score (Keep default) ---
    size_score = {
        "raspberry_pi": 0.6,
        "jetson_nano": 0.6,
        "desktop_pc": 0.9,
        "aws_server": 1.0,
    }

    # --- Compute Net Score ---
    net_score = compute_net_score({
        "ramp_up_time": ramp_up_time,
        "bus_factor": bus_factor,
        "license": license_score,
        "performance_claims": performance_claims,
        "dataset_and_code_score": dataset_and_code_score,
        "dataset_quality": dataset_quality,
        "code_quality": code_quality,
        "size_score": size_score
    })

    elapsed = int((time.time() - start) * 1000)

    # Return structure matching your pydantic models
    result = {
        "name": model_name,
        "category": "MODEL",
        "net_score": net_score,
        "ramp_up_time": ramp_up_time,
        "bus_factor": bus_factor,
        "performance_claims": performance_claims,
        "license": license_score,
        "dataset_and_code_score": dataset_and_code_score,
        "dataset_quality": dataset_quality,
        "code_quality": code_quality,
        "size_score": size_score,
        # Phase 2 Defaults
        "phase2": {
            "reproducibility": 0.5,
            "reviewedness": 0.5,
            "treescale_score": 0.5,
            "latency_ms": elapsed
        }
    }

    # Add latencies
    for key in list(result.keys()):
        if key not in ("name", "category", "size_score", "phase2"):
            result[f"{key}_latency"] = elapsed

    return result