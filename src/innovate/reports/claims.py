"""Claim taxonomy and claim-safety metadata for decision reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

REPORT_SCHEMA_VERSION = "1.0"
ClaimType = Literal["descriptive", "predictive", "simulation", "causal"]
CLAIM_TYPES: tuple[ClaimType, ...] = ("descriptive", "predictive", "simulation", "causal")

# Public wording that must not appear as unsupported recommendations.
FORBIDDEN_RECOMMENDATION_PHRASES: tuple[str, ...] = (
    "you should implement",
    "must adopt",
    "guaranteed to",
    "clinically indicated",
    "legally required",
    "regulatory approval",
)

SAFE_INTERPRETATION_BY_CLAIM: dict[ClaimType, str] = {
    "descriptive": (
        "Describes observed or fitted patterns under the stated data and model. "
        "Does not imply future performance or causal effects."
    ),
    "predictive": (
        "Projects outcomes under the fitted model and fixed assumptions. "
        "Predictions are not causal effects and are sensitive to extrapolation."
    ),
    "simulation": (
        "Summarizes counterfactual trajectories under an explicit simulation design. "
        "Results depend on structural assumptions and scenario inputs."
    ),
    "causal": (
        "Claims causal interpretation only when identification assumptions are stated "
        "and supported. Sensitivity analysis does not replace identification."
    ),
}


def classify_claim(claim_type: str) -> ClaimType:
    """Validate and return a canonical claim type."""
    normalized = str(claim_type).strip().lower()
    if normalized not in CLAIM_TYPES:
        raise ValueError(f"unsupported claim type '{claim_type}'; allowed={list(CLAIM_TYPES)}")
    return normalized  # type: ignore[return-value]


def assert_safe_public_wording(text: str) -> None:
    """Fail closed when public wording implies unsupported recommendations."""
    lowered = text.lower()
    for phrase in FORBIDDEN_RECOMMENDATION_PHRASES:
        if phrase in lowered:
            raise ValueError(f"public wording contains unsupported recommendation language: '{phrase}'")


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """A single claim with safety metadata for report consumers."""

    claim_type: ClaimType
    statement: str
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    allowed_as_recommendation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_type", classify_claim(self.claim_type))
        if not self.statement.strip():
            raise ValueError("statement must be non-empty")
        assert_safe_public_wording(self.statement)
        if self.allowed_as_recommendation:
            raise ValueError("allowed_as_recommendation must be False; automated recommendations are out of scope")
        assumptions = tuple(str(item).strip() for item in self.assumptions if str(item).strip())
        limitations = tuple(str(item).strip() for item in self.limitations if str(item).strip())
        for text in (*assumptions, *limitations):
            assert_safe_public_wording(text)
        # Causal claims must state identification assumptions (fail closed).
        if self.claim_type == "causal" and not assumptions:
            raise ValueError(
                "causal claims require at least one identification assumption; "
                "sensitivity alone does not establish causality"
            )
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_refs", tuple(str(item) for item in self.evidence_refs if str(item).strip()))

    @property
    def interpretation(self) -> str:
        return SAFE_INTERPRETATION_BY_CLAIM[self.claim_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "statement": self.statement,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "evidence_refs": list(self.evidence_refs),
            "allowed_as_recommendation": self.allowed_as_recommendation,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClaimRecord:
        return cls(
            claim_type=classify_claim(str(data["claim_type"])),
            statement=str(data["statement"]),
            assumptions=tuple(data.get("assumptions", ())),
            limitations=tuple(data.get("limitations", ())),
            evidence_refs=tuple(data.get("evidence_refs", ())),
            allowed_as_recommendation=bool(data.get("allowed_as_recommendation", False)),
        )


def claim_safety_summary(claims: Sequence[ClaimRecord]) -> dict[str, Any]:
    """Aggregate claim-safety metadata for release evidence and exports."""
    counts = dict.fromkeys(CLAIM_TYPES, 0)
    for claim in claims:
        counts[claim.claim_type] += 1
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "claim_counts": counts,
        "contains_causal_claims": counts["causal"] > 0,
        "recommendations_allowed": False,
        "forbidden_phrase_policy": "fail_closed",
    }
