"""JSON and Markdown export for decision reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from innovate.reports.claims import ClaimRecord
from innovate.reports.schemas import DecisionReport, build_decision_report


def export_report_json(report: DecisionReport, *, indent: int | None = 2) -> str:
    """Serialize a decision report to stable JSON text.

    Uses ``allow_nan=False`` so non-finite floats cannot leak invalid JSON.
    """
    return json.dumps(report.to_dict(), sort_keys=True, indent=indent, allow_nan=False)


def export_report_markdown(report: DecisionReport) -> str:
    """Render a decision report as Markdown with claim-safety language."""
    payload = report.to_dict()
    lines: list[str] = [
        f"# {payload['title']}",
        "",
        f"**Workflow:** `{payload['workflow']}`  ",
        f"**Schema:** `{payload['schema_version']}`  ",
        "**Recommendations:** not generated (out of scope)",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Claims",
        "",
    ]
    for claim in payload["claims"]:
        lines.extend(
            [
                f"### {claim['claim_type'].title()} claim",
                "",
                claim["statement"],
                "",
                f"*Interpretation boundary:* {claim['interpretation']}",
                "",
            ]
        )
        if claim["assumptions"]:
            lines.append("**Assumptions**")
            lines.extend(f"- {item}" for item in claim["assumptions"])
            lines.append("")
        if claim["limitations"]:
            lines.append("**Limitations**")
            lines.extend(f"- {item}" for item in claim["limitations"])
            lines.append("")

    if payload["assumptions"]:
        lines.extend(["## Report assumptions", ""])
        lines.extend(f"- {item}" for item in payload["assumptions"])
        lines.append("")

    if payload["limitations"]:
        lines.extend(["## Limitations", ""])
        lines.extend(f"- {item}" for item in payload["limitations"])
        lines.append("")

    if payload["diagnostics"]:
        lines.extend(
            ["## Diagnostics", "", "```json", json.dumps(payload["diagnostics"], sort_keys=True, indent=2), "```", ""]
        )
    if payload["uncertainty"]:
        lines.extend(
            ["## Uncertainty", "", "```json", json.dumps(payload["uncertainty"], sort_keys=True, indent=2), "```", ""]
        )
    if payload["sensitivity"]:
        lines.extend(
            ["## Sensitivity", "", "```json", json.dumps(payload["sensitivity"], sort_keys=True, indent=2), "```", ""]
        )
    if payload["explainability"]:
        lines.extend(
            [
                "## Explainability",
                "",
                "```json",
                json.dumps(payload["explainability"], sort_keys=True, indent=2),
                "```",
                "",
            ]
        )

    safety = payload["claim_safety"]
    lines.extend(
        [
            "## Claim safety",
            "",
            f"- Causal claims present: `{safety['contains_causal_claims']}`",
            f"- Recommendations allowed: `{safety['recommendations_allowed']}`",
            f"- Forbidden-phrase policy: `{safety['forbidden_phrase_policy']}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def example_policy_report() -> DecisionReport:
    """Deterministic example decision report for a policy workflow."""
    return build_decision_report(
        title="Policy intervention decision report (example)",
        workflow="policy",
        claims=(
            ClaimRecord(
                claim_type="simulation",
                statement="Under the stated timing scenario, cumulative adoption rises relative to baseline.",
                assumptions=("Fixed market size", "Intervention effect is additive"),
                limitations=("Simulation is not an empirical trial"),
            ),
            ClaimRecord(
                claim_type="predictive",
                statement="Model projections remain sensitive to the intervention effect parameter.",
                assumptions=("Stationary parameters outside the intervention"),
            ),
        ),
        assumptions=("Synthetic demo inputs only",),
        limitations=("Not legal, clinical, or regulatory advice",),
        sensitivity={"kind": "example", "note": "Attach live sensitivity blocks in production use"},
        explainability={"kind": "example", "note": "Attach live explainability blocks in production use"},
        metadata={"example": True, "workflow_family": "policy"},
    )


def example_competition_report() -> DecisionReport:
    return build_decision_report(
        title="Competition decision report (example)",
        workflow="competition",
        claims=(
            ClaimRecord(
                claim_type="descriptive",
                statement="Focal product share is lower than the strongest competitor under the fitted panel.",
            ),
        ),
        limitations=("Descriptive only; no causal claim about competitor actions"),
        metadata={"example": True, "workflow_family": "competition"},
    )


def example_substitution_report() -> DecisionReport:
    return build_decision_report(
        title="Substitution decision report (example)",
        workflow="substitution",
        claims=(
            ClaimRecord(
                claim_type="predictive",
                statement="The fitted substitution path is projected to cross the 50% share threshold.",
                assumptions=("Share path follows the fitted substitution model"),
            ),
        ),
        limitations=("Threshold crossings are model-based, not market guarantees"),
        metadata={"example": True, "workflow_family": "substitution"},
    )


def export_examples() -> dict[str, dict[str, str]]:
    """Return JSON and Markdown exports for built-in example reports."""
    reports: dict[str, DecisionReport] = {
        "policy": example_policy_report(),
        "competition": example_competition_report(),
        "substitution": example_substitution_report(),
    }
    return {
        name: {
            "json": export_report_json(report),
            "markdown": export_report_markdown(report),
        }
        for name, report in reports.items()
    }


def report_from_mapping(data: Mapping[str, Any]) -> DecisionReport:
    """Deserialize a decision report mapping."""
    return DecisionReport.from_dict(data)
