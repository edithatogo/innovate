"""Tests for claim taxonomy, sensitivity, explainability, and decision reports."""

from __future__ import annotations

import json

import pytest

from innovate.reports import (
    AssumptionSensitivityInput,
    ClaimRecord,
    DecisionReport,
    ParameterSensitivityInput,
    ThresholdSensitivityInput,
    TimingSensitivityInput,
    adoption_driver_summary,
    assumption_sensitivity_summary,
    build_decision_report,
    classify_claim,
    combine_explainability,
    combine_sensitivity_summaries,
    competition_effect_summary,
    example_competition_report,
    example_policy_report,
    example_substitution_report,
    export_examples,
    export_report_json,
    export_report_markdown,
    intervention_timing_summary,
    parameter_perturbation_summary,
    policy_component_summary,
    substitution_threshold_summary,
    threshold_sensitivity_summary,
)


def test_claim_taxonomy_classification() -> None:
    assert classify_claim("Descriptive") == "descriptive"
    assert classify_claim("PREDICTIVE") == "predictive"
    assert classify_claim("simulation") == "simulation"
    assert classify_claim("causal") == "causal"
    with pytest.raises(ValueError, match="unsupported claim type"):
        classify_claim("prescriptive")


def test_claim_record_rejects_recommendation_language() -> None:
    with pytest.raises(ValueError, match="unsupported recommendation"):
        ClaimRecord(claim_type="predictive", statement="You should implement this policy now.")
    with pytest.raises(ValueError, match="allowed_as_recommendation"):
        ClaimRecord(
            claim_type="descriptive",
            statement="Share increased in the sample window.",
            allowed_as_recommendation=True,
        )


def test_decision_report_envelope_and_claim_safety() -> None:
    report = build_decision_report(
        title="Demo",
        workflow="policy",
        claims=(
            ClaimRecord(claim_type="descriptive", statement="Adoption rose in the fitted window."),
            ClaimRecord(
                claim_type="causal",
                statement="Under stated identification assumptions, the intervention raised adoption.",
                assumptions=("No unmeasured confounding",),
                limitations=("Assumptions are not proven"),
            ),
        ),
        assumptions=("Synthetic demo",),
        limitations=("Not regulatory advice",),
    )
    safety = report.claim_safety()
    assert safety["contains_causal_claims"] is True
    assert safety["recommendations_allowed"] is False
    payload = report.to_dict()
    restored = DecisionReport.from_dict(payload)
    assert restored.title == "Demo"
    assert "causal" in restored.recommended_interpretation().lower()


def test_parameter_perturbation_and_elasticity_deterministic() -> None:
    def outcome(params):
        return 10.0 + 2.0 * params["beta"]

    summary = parameter_perturbation_summary(
        outcome,
        [ParameterSensitivityInput(name="beta", baseline=1.0, deltas=(-0.1, 0.1))],
    )
    assert summary["deterministic"] is True
    assert len(summary["rows"]) == 2
    again = parameter_perturbation_summary(
        outcome,
        [ParameterSensitivityInput(name="beta", baseline=1.0, deltas=(-0.1, 0.1))],
    )
    assert summary == again
    # elasticity = (dY/Y) / (dX/X) = (0.2/12) / (0.1/1) ≈ 0.1667 for +0.1
    plus = next(row for row in summary["rows"] if row["delta"] == 0.1)
    assert plus["absolute_change"] == pytest.approx(0.2)
    assert plus["elasticity"] == pytest.approx((0.2 / 12.0) / 0.1)


def test_assumption_timing_threshold_sensitivity() -> None:
    def outcome(params):
        return params.get("alpha", 1.0) * (10.0 - 0.5 * params.get("t_event", 0.0))

    assumption = assumption_sensitivity_summary(
        outcome,
        [AssumptionSensitivityInput(name="alpha", baseline=1.0, alternatives=(0.5, 1.5))],
        context={"t_event": 2.0},
    )
    timing = intervention_timing_summary(
        outcome,
        [TimingSensitivityInput(name="t_event", baseline_time=2.0, offsets=(-1.0, 1.0))],
        context={"alpha": 1.0},
    )
    threshold = threshold_sensitivity_summary(
        [1.0, 2.0, 3.0, 4.0],
        [ThresholdSensitivityInput(name="cut", thresholds=(2.5, 3.5))],
    )
    combined = combine_sensitivity_summaries(assumption, timing, threshold)
    assert combined["deterministic"] is True
    assert len(combined["blocks"]) == 3
    assert threshold["rows"][0]["n_meet"] == 2


def test_explainability_summaries() -> None:
    drivers = adoption_driver_summary(
        {"price": 2.0, "network": 1.0, "awareness": 1.0},
        baseline_adoption=10.0,
        scenario_adoption=18.0,
    )
    assert drivers["delta"] == 8.0
    assert abs(sum(item["weight"] for item in drivers["contributions"].values()) - 1.0) < 1e-9

    competition = competition_effect_summary({"A": 0.4, "B": 0.35, "C": 0.25}, focal_product="A")
    assert competition["competitive_pressure"] == pytest.approx(0.6)
    assert competition["lead"] == pytest.approx(0.05)

    substitution = substitution_threshold_summary([0.05, 0.2, 0.4, 0.55], thresholds=(0.25, 0.5))
    assert substitution["crossings"][0]["first_index"] == 2
    assert substitution["crossings"][1]["first_index"] == 3

    policy = policy_component_summary({"subsidy": 0.3, "mandate": 0.1}, total_effect=0.5)
    assert policy["dominant_component"] == "subsidy"
    assert policy["residual"] == pytest.approx(0.1)
    combined = combine_explainability(drivers, competition, substitution, policy)
    assert combined["deterministic"] is True


def test_json_and_markdown_export_examples() -> None:
    for builder in (example_policy_report, example_competition_report, example_substitution_report):
        report = builder()
        text = export_report_json(report)
        md = export_report_markdown(report)
        payload = json.loads(text)
        assert payload["schema_version"] == "1.0"
        assert payload["claim_safety"]["recommendations_allowed"] is False
        assert report.title in md
        assert "Claim safety" in md
        assert "not generated" in md.lower()
    examples = export_examples()
    assert set(examples) == {"policy", "competition", "substitution"}
    for bundle in examples.values():
        assert "json" in bundle
        assert "markdown" in bundle
