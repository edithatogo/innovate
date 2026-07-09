"""Decision-report envelopes with assumptions, diagnostics, and uncertainty."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from innovate.reports.claims import (
    REPORT_SCHEMA_VERSION,
    ClaimRecord,
    claim_safety_summary,
)

WorkflowFamily = Literal["policy", "competition", "substitution", "diffusion", "generic"]


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """Stable decision-report artifact for researchers and policy analysts.

    Does not produce automated policy/legal/clinical recommendations.
    """

    title: str
    workflow: WorkflowFamily
    claims: tuple[ClaimRecord, ...]
    assumptions: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    sensitivity: Mapping[str, Any] = field(default_factory=dict)
    explainability: Mapping[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    schema_version: str = REPORT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if self.workflow not in {"policy", "competition", "substitution", "diffusion", "generic"}:
            raise ValueError(f"unsupported workflow: {self.workflow}")
        if not self.claims:
            raise ValueError("claims must be non-empty")
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported report schema_version: {self.schema_version}")
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "uncertainty", dict(self.uncertainty))
        object.__setattr__(self, "sensitivity", dict(self.sensitivity))
        object.__setattr__(self, "explainability", dict(self.explainability))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.interpretation:
            from innovate.reports.claims import assert_safe_public_wording

            assert_safe_public_wording(self.interpretation)

    def claim_safety(self) -> dict[str, Any]:
        return claim_safety_summary(self.claims)

    def recommended_interpretation(self) -> str:
        if self.interpretation.strip():
            return self.interpretation
        # Deterministic default from claim mix.
        types = sorted({claim.claim_type for claim in self.claims})
        if "causal" in types:
            return (
                "Interpret causal statements only under the listed identification assumptions. "
                "Sensitivity results describe robustness, not proof of causality."
            )
        if "simulation" in types:
            return (
                "Interpret results as simulation outcomes under explicit scenario inputs. "
                "Do not treat them as empirical policy evaluations without further design."
            )
        if "predictive" in types:
            return "Interpret results as model-based projections under fixed assumptions. They are not causal effects."
        return (
            "Interpret results as descriptive summaries of fitted or observed patterns. "
            "They do not authorize automated recommendations."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "workflow": self.workflow,
            "claims": [claim.to_dict() for claim in self.claims],
            "assumptions": list(self.assumptions),
            "diagnostics": dict(self.diagnostics),
            "uncertainty": dict(self.uncertainty),
            "limitations": list(self.limitations),
            "sensitivity": dict(self.sensitivity),
            "explainability": dict(self.explainability),
            "interpretation": self.recommended_interpretation(),
            "claim_safety": self.claim_safety(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionReport:
        claims = tuple(ClaimRecord.from_dict(item) for item in data["claims"])
        return cls(
            title=str(data["title"]),
            workflow=data["workflow"],  # validated in __post_init__
            claims=claims,
            assumptions=tuple(data.get("assumptions", ())),
            diagnostics=dict(data.get("diagnostics", {})),
            uncertainty=dict(data.get("uncertainty", {})),
            limitations=tuple(data.get("limitations", ())),
            sensitivity=dict(data.get("sensitivity", {})),
            explainability=dict(data.get("explainability", {})),
            interpretation=str(data.get("interpretation", "")),
            schema_version=str(data.get("schema_version", REPORT_SCHEMA_VERSION)),
            metadata=dict(data.get("metadata", {})),
        )


def build_decision_report(
    *,
    title: str,
    workflow: WorkflowFamily,
    claims: Sequence[ClaimRecord],
    assumptions: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
    sensitivity: Mapping[str, Any] | None = None,
    explainability: Mapping[str, Any] | None = None,
    interpretation: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> DecisionReport:
    """Construct a validated decision report envelope."""
    return DecisionReport(
        title=title,
        workflow=workflow,
        claims=tuple(claims),
        assumptions=tuple(assumptions),
        diagnostics=dict(diagnostics or {}),
        uncertainty=dict(uncertainty or {}),
        limitations=tuple(limitations),
        sensitivity=dict(sensitivity or {}),
        explainability=dict(explainability or {}),
        interpretation=interpretation,
        metadata=dict(metadata or {}),
    )
