"""
License information models for artifact licensing.

This module defines the license structure and compatibility
evaluation for artifacts, supporting SPDX license identifiers
and various license sources.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


# Common SPDX license identifiers
PERMISSIVE_LICENSES = frozenset({
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "CC0-1.0",
    "WTFPL",
})

COPYLEFT_LICENSES = frozenset({
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "MPL-2.0",
})

SPECIAL_LICENSES = frozenset({
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-ND-4.0",
    "CC-BY-NC-ND-4.0",
})

# License source types
LicenseSource = Literal["hf", "github", "manual", "unknown"]


class LicenseInfo(BaseModel):
    """
    License information for an artifact.
    
    Stores normalized license data including the SPDX identifier,
    source of the license information, and compatibility status.
    """
    model_config = ConfigDict(extra="forbid")
    
    license_id: str = Field(
        default="UNKNOWN",
        description="SPDX license identifier (e.g., 'MIT', 'Apache-2.0')"
    )
    license_source: str = Field(
        default="unknown",
        description="Source of license info: 'hf', 'github', 'manual', 'unknown'"
    )
    license_compatible: bool = Field(
        default=False,
        description="Whether the license is compatible with intended use"
    )
    license_url: Optional[str] = Field(
        default=None,
        description="URL to the full license text"
    )
    license_text: Optional[str] = Field(
        default=None,
        max_length=65536,
        description="Full license text (truncated if too long)"
    )

    @field_validator("license_source", mode="before")
    @classmethod
    def normalize_source(cls, v: Any) -> str:
        """Normalize the license source to allowed values."""
        if v is None:
            return "unknown"
        v_lower = str(v).lower()
        if v_lower in ("hf", "huggingface", "hugging_face"):
            return "hf"
        elif v_lower in ("github", "gh"):
            return "github"
        elif v_lower == "manual":
            return "manual"
        return "unknown"

    @field_validator("license_id", mode="before")
    @classmethod
    def normalize_license_id(cls, v: Any) -> str:
        """Normalize the license ID."""
        if v is None or v == "":
            return "UNKNOWN"
        return str(v).strip()

    @property
    def is_permissive(self) -> bool:
        """Check if the license is permissive."""
        return self.license_id in PERMISSIVE_LICENSES

    @property
    def is_copyleft(self) -> bool:
        """Check if the license is copyleft."""
        return self.license_id in COPYLEFT_LICENSES

    @property
    def is_known(self) -> bool:
        """Check if the license is a known SPDX identifier."""
        return self.license_id in (PERMISSIVE_LICENSES | COPYLEFT_LICENSES | SPECIAL_LICENSES)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for DynamoDB storage.
        
        Returns:
            Dictionary representation of license info
        """
        result = {
            "license_id": self.license_id,
            "license_source": self.license_source,
            "license_compatible": self.license_compatible,
        }
        if self.license_url:
            result["license_url"] = self.license_url
        # Omit license_text from storage to save space
        return result

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "LicenseInfo":
        """
        Create from dictionary (e.g., from DynamoDB).
        
        Args:
            data: Dictionary with license fields
            
        Returns:
            LicenseInfo instance
        """
        if not data:
            return cls()
        return cls(**data)


def evaluate_license_compatibility(
    license_info: LicenseInfo,
    target_license: Optional[str] = None,
    allow_copyleft: bool = False,
) -> bool:
    """
    Evaluate if a license is compatible with intended use.
    
    This is a placeholder implementation that will be extended
    with more sophisticated license compatibility logic.
    
    Args:
        license_info: The license information to evaluate
        target_license: Optional target license to check compatibility with
        allow_copyleft: Whether copyleft licenses are acceptable
        
    Returns:
        True if the license is compatible, False otherwise
    """
    # Unknown licenses are not compatible by default
    if license_info.license_id == "UNKNOWN":
        return False
    
    # Permissive licenses are always compatible
    if license_info.is_permissive:
        return True
    
    # Copyleft licenses depend on allow_copyleft flag
    if license_info.is_copyleft:
        return allow_copyleft
    
    # Special licenses (like CC) need case-by-case evaluation
    # For now, mark as compatible if known
    if license_info.is_known:
        return True
    
    # Unknown license types are not compatible
    return False


def check_license_chain_compatibility(
    artifact_license: LicenseInfo,
    parent_licenses: list[LicenseInfo],
) -> tuple[bool, list[str]]:
    """
    Check if an artifact's license is compatible with its parents.
    
    Evaluates the license compatibility across the artifact's
    lineage to ensure no license conflicts.
    
    Args:
        artifact_license: The artifact's license
        parent_licenses: List of parent artifacts' licenses
        
    Returns:
        Tuple of (is_compatible, list_of_incompatible_license_ids)
    """
    incompatible = []
    
    for parent_license in parent_licenses:
        # Skip unknown licenses in parents
        if parent_license.license_id == "UNKNOWN":
            continue
        
        # Copyleft parent + non-copyleft child is problematic
        if parent_license.is_copyleft and not artifact_license.is_copyleft:
            # This is a potential violation (simplified check)
            incompatible.append(parent_license.license_id)
    
    return len(incompatible) == 0, incompatible
