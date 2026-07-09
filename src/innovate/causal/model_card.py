"""Model card and release evidence for causal analyses.

This module provides tools for documenting causal model assumptions,
limitations, and release evidence for policy claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CausalModelCard:
    """Model card for causal analysis.

    Attributes
    ----------
        name: Name of the causal analysis
        description: Description of the analysis purpose
        estimand: Type of estimand (ATE, ATT, CATE)
        assumptions: List of identifying assumptions
        limitations: Known limitations and caveats
        data_sources: List of data sources used
        date_created: When the analysis was created
        version: Version of the analysis
    """

    name: str
    description: str
    estimand: str
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    date_created: str | None = None
    version: str = "1.0"

    def __post_init__(self):
        """Set creation date if not provided."""
        if self.date_created is None:
            self.date_created = datetime.now().isoformat()

    def add_assumption(self, assumption: str) -> None:
        """Add an identifying assumption."""
        self.assumptions.append(assumption)

    def add_limitation(self, limitation: str) -> None:
        """Add a known limitation."""
        self.limitations.append(limitation)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "estimand": self.estimand,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "data_sources": self.data_sources,
            "date_created": self.date_created,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ReleaseEvidence:
    """Evidence and caveats for releasing causal claim.

    Attributes
    ----------
        claim: The causal claim being made
        supporting_evidence: List of evidence items supporting the claim
        caveats: List of important caveats
        sensitivity_analysis_conducted: Whether sensitivity analysis was done
        evidence_level: Confidence level (high, medium, low)
        approved_for_release: Whether approved for external communication
    """

    claim: str
    supporting_evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    sensitivity_analysis_conducted: bool = False
    evidence_level: str = "medium"
    approved_for_release: bool = False

    def add_evidence(self, evidence: str) -> None:
        """Add supporting evidence."""
        self.supporting_evidence.append(evidence)

    def add_caveat(self, caveat: str) -> None:
        """Add a caveat or limitation."""
        self.caveats.append(caveat)

    def validate_for_release(self) -> tuple[bool, list[str]]:
        """Validate that evidence is sufficient for release.

        Returns
        -------
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not self.supporting_evidence:
            issues.append("No supporting evidence provided")

        if not self.caveats:
            issues.append("No caveats documented")

        if self.evidence_level == "low":
            issues.append("Evidence level is low - requires external review")

        if not self.sensitivity_analysis_conducted and self.evidence_level == "medium":
            issues.append("Sensitivity analysis recommended for medium-confidence claims")

        is_valid = len(issues) == 0
        self.approved_for_release = is_valid

        return is_valid, issues

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim,
            "supporting_evidence": self.supporting_evidence,
            "caveats": self.caveats,
            "sensitivity_analysis_conducted": self.sensitivity_analysis_conducted,
            "evidence_level": self.evidence_level,
            "approved_for_release": self.approved_for_release,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AssumptionDocument:
    """Document identifying assumptions and their justification.

    Attributes
    ----------
        assumption_name: Name of the assumption
        mathematical_statement: Mathematical formulation
        intuitive_explanation: Plain English explanation
        how_checked: How the assumption was assessed
        sensitivity_to_violation: How robust results are to violation
    """

    assumption_name: str
    mathematical_statement: str
    intuitive_explanation: str
    how_checked: str | None = None
    sensitivity_to_violation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assumption_name": self.assumption_name,
            "mathematical_statement": self.mathematical_statement,
            "intuitive_explanation": self.intuitive_explanation,
            "how_checked": self.how_checked,
            "sensitivity_to_violation": self.sensitivity_to_violation,
        }

    def to_markdown(self) -> str:
        """Convert to markdown for documentation."""
        md = f"### {self.assumption_name}\n\n"
        md += f"**Mathematical Statement:** {self.mathematical_statement}\n\n"
        md += f"**Intuitive Explanation:** {self.intuitive_explanation}\n\n"
        if self.how_checked:
            md += f"**How Checked:** {self.how_checked}\n\n"
        if self.sensitivity_to_violation:
            md += f"**Sensitivity to Violation:** {self.sensitivity_to_violation}\n\n"
        return md
