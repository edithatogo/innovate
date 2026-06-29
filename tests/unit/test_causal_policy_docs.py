"""Tests for causal policy documentation and model cards.

This module tests documentation generation, model card creation,
and release evidence validation.
"""

from __future__ import annotations

import pytest

from innovate.causal.model_card import (
    AssumptionDocument,
    CausalModelCard,
    ReleaseEvidence,
)


class TestCausalModelCard:
    """Test model card creation and validation."""

    def test_basic_model_card(self):
        """Test creating a basic model card."""
        card = CausalModelCard(
            name="policy_impact_ate",
            description="Estimates average effect of policy intervention",
            estimand="ATE",
        )
        assert card.name == "policy_impact_ate"
        assert card.estimand == "ATE"
        assert card.date_created is not None

    def test_model_card_with_assumptions(self):
        """Test model card with documented assumptions."""
        card = CausalModelCard(
            name="heterogeneous_effects",
            description="Heterogeneous effects by demographic group",
            estimand="CATE",
        )
        card.add_assumption("Unconfoundedness: all confounders are observed")
        card.add_assumption("Overlap: 0 < P(T=1|X) < 1 for all X")
        card.add_assumption("No interference between units")

        assert len(card.assumptions) == 3
        assert "Unconfoundedness" in card.assumptions[0]

    def test_model_card_with_limitations(self):
        """Test model card with documented limitations."""
        card = CausalModelCard(
            name="limited_model",
            description="Model with known limitations",
            estimand="ATT",
        )
        card.add_limitation("Small sample size (n=50 treated)")
        card.add_limitation("Missing data on key confounder (income)")
        card.add_limitation("Analysis limited to 2020-2021 period")

        assert len(card.limitations) == 3

    def test_model_card_json_export(self):
        """Test exporting model card to JSON."""
        card = CausalModelCard(
            name="export_test",
            description="Test JSON export",
            estimand="ATE",
        )
        card.add_assumption("Assumption 1")
        card.add_limitation("Limitation 1")

        json_str = card.to_json()
        assert "export_test" in json_str
        assert "Assumption 1" in json_str


class TestReleaseEvidence:
    """Test release evidence and approval workflow."""

    def test_release_evidence_basic(self):
        """Test creating release evidence."""
        evidence = ReleaseEvidence(claim="Policy intervention increases adoption by 15%")
        assert evidence.claim is not None
        assert not evidence.approved_for_release

    def test_release_evidence_validation_passes(self):
        """Test validation passes with complete evidence."""
        evidence = ReleaseEvidence(
            claim="Treatment effect is significant",
            supporting_evidence=[
                "Difference-in-differences: +0.15 [95% CI: 0.10-0.20]",
                "Event-study parallel trends test: p-value = 0.45",
            ],
            caveats=[
                "Assumes no unmeasured confounding",
                "Limited to 2020-2025 period",
            ],
            evidence_level="high",
            sensitivity_analysis_conducted=True,
        )

        is_valid, issues = evidence.validate_for_release()
        assert is_valid
        assert len(issues) == 0
        assert evidence.approved_for_release

    def test_release_evidence_validation_fails(self):
        """Test validation fails with insufficient evidence."""
        evidence = ReleaseEvidence(
            claim="Treatment effect exists",
            evidence_level="low",
        )

        is_valid, issues = evidence.validate_for_release()
        assert not is_valid
        assert len(issues) > 0
        assert any("evidence" in issue.lower() for issue in issues)

    def test_release_evidence_with_caveats(self):
        """Test that strong claims require sensitivity analysis."""
        evidence = ReleaseEvidence(
            claim="Causal effect identified with no confounding",
            supporting_evidence=["Test statistic: t=5.2"],
            caveats=["Potential unmeasured confounding"],
            evidence_level="medium",
            sensitivity_analysis_conducted=False,
        )

        is_valid, issues = evidence.validate_for_release()
        assert not is_valid
        assert any("sensitivity" in issue.lower() for issue in issues)


class TestAssumptionDocument:
    """Test assumption documentation for transparency."""

    def test_unconfoundedness_assumption(self):
        """Test documenting unconfoundedness assumption."""
        assume = AssumptionDocument(
            assumption_name="Unconfoundedness",
            mathematical_statement="(Y_0, Y_1) ⊥ T | X",
            intuitive_explanation=("All variables that affect both treatment and outcome are observed"),
            how_checked="Sensitivity analysis with E-values",
        )

        assert "Unconfoundedness" in assume.assumption_name
        d = assume.to_dict()
        assert d["assumption_name"] == "Unconfoundedness"

    def test_overlap_assumption(self):
        """Test documenting overlap assumption."""
        assume = AssumptionDocument(
            assumption_name="Positivity (Overlap)",
            mathematical_statement="0 < P(T=1|X) < 1 for all X in support",
            intuitive_explanation=("All units have positive probability of being treated and untreated"),
            how_checked="Check propensity score distribution [0, 1]",
            sensitivity_to_violation=("Lack of overlap creates off-support inference problems"),
        )

        md = assume.to_markdown()
        assert "Positivity" in md
        assert "Mathematical Statement:" in md

    def test_assumption_markdown_output(self):
        """Test markdown output for documentation."""
        assume = AssumptionDocument(
            assumption_name="Test Assumption",
            mathematical_statement="Y ⊥ T | X",
            intuitive_explanation="Test explanation",
            how_checked="Test check",
        )

        md = assume.to_markdown()
        assert "### Test Assumption" in md
        assert "Mathematical Statement:" in md
        assert "How Checked:" in md


class TestDocumentationExamples:
    """Test that documentation examples work correctly."""

    def test_complete_model_card_example(self):
        """Test a complete, realistic model card."""
        card = CausalModelCard(
            name="policy_diffusion_ate",
            description=("Average treatment effect of policy incentive on technology adoption"),
            estimand="ATE",
            data_sources=[
                "Administrative data from 2020-2025",
                "Survey responses (2021-2022)",
            ],
        )

        # Document assumptions
        card.add_assumption("Unconfoundedness: all confounders are observed")
        card.add_assumption("Overlap: positive probability for all units")
        card.add_assumption("No network effects/spillovers")
        card.add_assumption("Stable unit treatment value assumption (SUTVA)")

        # Document limitations
        card.add_limitation("Missing data on pre-policy adoption (pre-2020)")
        card.add_limitation("Small subgroups (n<30) have wide confidence intervals")
        card.add_limitation("No measurement error adjustment")
        card.add_limitation("Analysis limited to single country")

        card_dict = card.to_dict()
        assert len(card_dict["assumptions"]) == 4
        assert len(card_dict["limitations"]) == 4
        assert card_dict["estimand"] == "ATE"

    def test_release_workflow_example(self):
        """Test complete release workflow."""
        # Create analysis evidence
        evidence = ReleaseEvidence(
            claim=("Policy increased adoption rates by approximately 18 percentage points"),
            supporting_evidence=[
                "Difference-in-differences estimate: 0.18 [95% CI: 0.13-0.23]",
                "Event-study analysis shows effects sustained over 3-year period",
                "Covariate balance: all SMDs < 0.05 after matching",
                "Parallel trends test: pre-treatment coefficient = 0.01 (p=0.67)",
            ],
            caveats=[
                "Analysis assumes no unmeasured confounding",
                "Limited to jurisdictions that adopted policy",
                "No long-term follow-up beyond 3 years",
            ],
            sensitivity_analysis_conducted=True,
            evidence_level="high",
        )

        # Validate before release
        is_valid, issues = evidence.validate_for_release()
        assert is_valid
        assert evidence.approved_for_release

    def test_assumption_documentation_for_policy_brief(self):
        """Test documenting assumptions for policy brief."""
        assumptions = [
            AssumptionDocument(
                assumption_name="Unconfoundedness (Selection on Observables)",
                mathematical_statement="(Y₀, Y₁) ⊥ T | X",
                intuitive_explanation=(
                    "After controlling for observed demographics, income, "
                    "and prior adoption, treatment assignment is random"
                ),
                how_checked=("Covariate balance test: SMD < 0.1 for all covariates"),
                sensitivity_to_violation=(
                    "E-value = 1.8: unmeasured confounder would need 80% higher association to explain away effect"
                ),
            ),
            AssumptionDocument(
                assumption_name="Common Support (Overlap)",
                mathematical_statement="0 < P(T=1|X) < 1 ∀X ∈ Supp(X)",
                intuitive_explanation=(
                    "Both treated and untreated units exist for all observed "
                    "combinations of demographic and economic characteristics"
                ),
                how_checked=("Propensity score overlap check: treated range [0.15, 0.85], control range [0.12, 0.87]"),
                sensitivity_to_violation=(
                    "Off-support extrapolation can bias estimates; matched sample n=400 (84% of original)"
                ),
            ),
        ]

        # Convert to markdown for policy brief
        docs = "\n".join(a.to_markdown() for a in assumptions)
        assert "Unconfoundedness" in docs
        assert "Common Support" in docs
        assert "E-value" in docs
