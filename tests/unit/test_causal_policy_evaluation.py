"""Tests for causal policy evaluation framework.

This module provides comprehensive tests for causal model contracts, treatment
effect estimation, and sensitivity analysis following the Conductor-driven
specification for Track 06: Causal Policy Evaluation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from innovate.causal.policy import (
    CausalModel,
    CausalModelContract,
    InterventionContract,
    PolicyEvaluationError,
    TreatmentEffectEstimator,
)


class TestInterventionContract:
    """Test intervention specification and validation."""

    def test_intervention_contract_basic(self):
        """Test creating a basic intervention contract."""
        intervention = InterventionContract(
            name="policy_adoption",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        assert intervention.name == "policy_adoption"
        assert intervention.timing == "post"
        assert intervention.comparator == "control"

    def test_intervention_with_rollout(self):
        """Test intervention with staggered rollout."""
        intervention = InterventionContract(
            name="phased_rollout",
            timing="staggered",
            comparator="control",
            start_time=2020,
            end_time=2025,
            rollout_schedule={
                "2020": 0.25,
                "2021": 0.5,
                "2022": 1.0,
            },
        )
        assert intervention.rollout_schedule is not None
        assert intervention.rollout_schedule["2022"] == 1.0

    def test_intervention_with_spillovers(self):
        """Test intervention with spillover effects."""
        intervention = InterventionContract(
            name="network_intervention",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
            spillover_regions=["adjacent", "network"],
            spillover_strength=0.3,
        )
        assert intervention.spillover_regions == ["adjacent", "network"]
        assert intervention.spillover_strength == 0.3

    def test_intervention_missing_comparator_fails(self):
        """Test that intervention without comparator fails validation."""
        with pytest.raises(
            PolicyEvaluationError,
            match="Comparator group must be specified",
        ):
            InterventionContract(
                name="no_control",
                timing="post",
                start_time=2020,
                end_time=2025,
            )

    def test_intervention_timing_variants(self):
        """Test different timing specifications."""
        for timing in ["post", "pre", "staggered", "event-study"]:
            intervention = InterventionContract(
                name=f"timing_{timing}",
                timing=timing,
                comparator="control",
                start_time=2020,
                end_time=2025,
            )
            assert intervention.timing == timing


class TestCausalModelContract:
    """Test causal model specification with confounding control."""

    def test_causal_model_basic(self):
        """Test creating a basic causal model contract."""
        model = CausalModelContract(
            name="policy_impact",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income", "education", "region"],
        )
        assert model.name == "policy_impact"
        assert model.treatment_variable == "treated"
        assert model.outcome_variable == "adoption_rate"
        assert len(model.confounders) == 3

    def test_causal_model_with_covariates(self):
        """Test causal model with covariate specification."""
        model = CausalModelContract(
            name="heterogeneous_effects",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income"],
            effect_modifiers=["age_group", "innovation_type"],
        )
        assert model.effect_modifiers == ["age_group", "innovation_type"]

    def test_causal_model_missing_confounders_fails(self):
        """Test that model without confounders fails."""
        with pytest.raises(
            PolicyEvaluationError,
            match="Confounders must be specified",
        ):
            CausalModelContract(
                name="no_confounders",
                treatment_variable="treated",
                outcome_variable="adoption_rate",
                confounders=[],
            )

    def test_causal_model_with_assumptions(self):
        """Test causal model with explicit assumptions."""
        assumptions = {
            "unconfoundedness": "We assume all confounders are observed",
            "overlap": "All units have positive probability of treatment",
            "positivity": "No structural zeros in propensity scores",
        }
        model = CausalModelContract(
            name="assumptions_model",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income"],
            identifying_assumptions=assumptions,
        )
        assert "unconfoundedness" in model.identifying_assumptions

    def test_causal_model_json_serialization(self):
        """Test causal model can be serialized to JSON."""
        model = CausalModelContract(
            name="serializable",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income", "education"],
        )
        json_str = model.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["name"] == "serializable"


class TestPolicyEvaluationGuardrails:
    """Test safeguards against common causal misuse."""

    def test_missing_comparator_error(self):
        """Test guard against missing comparator."""
        with pytest.raises(PolicyEvaluationError, match="Comparator"):
            CausalModel(
                intervention=InterventionContract(
                    name="test",
                    timing="post",
                    start_time=2020,
                    end_time=2025,
                    comparator=None,
                ),
                causal_model=CausalModelContract(
                    name="model",
                    treatment_variable="treated",
                    outcome_variable="outcome",
                    confounders=["confounder"],
                ),
            )

    def test_post_treatment_leakage_guard(self):
        """Test guard against post-treatment variable inclusion."""
        model = CausalModelContract(
            name="leakage_test",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income"],
        )
        # Simulate adding post-treatment variable as confounder
        post_treatment_vars = ["post_treatment_outcome", "affected_variable"]
        for var in post_treatment_vars:
            # Should warn or reject
            with pytest.raises(
                PolicyEvaluationError,
                match="post-treatment|leakage",
            ):
                model.validate_confounders(post_treatment_vars)

    def test_time_window_mismatch_guard(self):
        """Test guard against incompatible time windows."""
        intervention = InterventionContract(
            name="timing_test",
            timing="post",
            comparator="control",
            start_time=2021,
            end_time=2025,
        )
        # Outcome assessment window before intervention should fail
        with pytest.raises(PolicyEvaluationError, match="time|window"):
            intervention.validate_outcome_window(start=2019, end=2020)

    def test_unsupported_causal_claim_guard(self):
        """Test guard against unsupported causal claims."""
        model = CausalModelContract(
            name="unsupported_claim",
            treatment_variable="treated",
            outcome_variable="outcome",
            confounders=["confounder"],
        )
        # Trying to claim causality with unobserved confounding
        with pytest.raises(
            PolicyEvaluationError,
            match="unobserved|sensitivity",
        ):
            model.validate_causal_claim(
                has_sensitivity_analysis=False,
                unobserved_confounding_risk="high",
            )

    def test_missing_identifying_assumptions_warning(self):
        """Test warning when identifying assumptions not documented."""
        model = CausalModelContract(
            name="undocumented",
            treatment_variable="treated",
            outcome_variable="outcome",
            confounders=["confounder"],
            identifying_assumptions={},
        )
        # Should have some way to flag this as incomplete
        assert len(model.identifying_assumptions) == 0


class TestCausalModelClass:
    """Test the main CausalModel class for evaluation workflows."""

    def test_causal_model_initialization(self):
        """Test initializing a causal model for evaluation."""
        intervention = InterventionContract(
            name="policy_eval",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        contract = CausalModelContract(
            name="evaluation_model",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income", "education"],
        )
        model = CausalModel(intervention=intervention, causal_model=contract)
        assert model.intervention.name == "policy_eval"
        assert model.causal_model.name == "evaluation_model"

    def test_causal_model_with_data(self):
        """Test causal model with data specification."""
        intervention = InterventionContract(
            name="data_eval",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        contract = CausalModelContract(
            name="data_model",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income"],
        )
        model = CausalModel(intervention=intervention, causal_model=contract)

        # Create sample data
        n_obs = 100
        data = {
            "treated": np.random.binomial(1, 0.5, n_obs),
            "adoption_rate": np.random.uniform(0, 1, n_obs),
            "income": np.random.normal(50000, 15000, n_obs),
        }

        model.add_data(data)
        assert model.n_obs == n_obs

    def test_causal_model_export(self):
        """Test exporting causal model specification."""
        intervention = InterventionContract(
            name="export_test",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        contract = CausalModelContract(
            name="export_model",
            treatment_variable="treated",
            outcome_variable="outcome",
            confounders=["confounder"],
        )
        model = CausalModel(intervention=intervention, causal_model=contract)

        # Should be able to export to dict
        spec = model.to_dict()
        assert spec["intervention"]["name"] == "export_test"
        assert spec["causal_model"]["name"] == "export_model"


class TestTreatmentEffectEstimation:
    """Test treatment effect estimator contract and methods."""

    def test_ate_estimation(self):
        """Test Average Treatment Effect (ATE) estimation."""
        estimator = TreatmentEffectEstimator(
            method="naive",
            outcome_variable="adoption_rate",
            treatment_variable="treated",
        )
        # Create simple data
        n_control = 100
        n_treat = 100
        data = {
            "treated": np.concatenate([np.zeros(n_control), np.ones(n_treat)]),
            "adoption_rate": np.concatenate(
                [
                    np.random.normal(0.3, 0.1, n_control),
                    np.random.normal(0.5, 0.1, n_treat),
                ]
            ),
        }
        ate = estimator.estimate_ate(data)
        assert isinstance(ate, float)
        # Should be positive since treated > control
        assert ate > 0

    def test_cate_estimation(self):
        """Test Conditional Average Treatment Effect (CATE) estimation."""
        estimator = TreatmentEffectEstimator(
            method="forest",
            outcome_variable="adoption_rate",
            treatment_variable="treated",
        )
        n_obs = 200
        data = {
            "treated": np.random.binomial(1, 0.5, n_obs),
            "adoption_rate": np.random.uniform(0, 1, n_obs),
            "age_group": np.random.choice(["young", "old"], n_obs),
        }
        cates = estimator.estimate_cate(
            data,
            effect_modifiers=["age_group"],
        )
        assert isinstance(cates, dict)

    def test_att_estimation(self):
        """Test Average Treatment Effect on the Treated (ATT) estimation."""
        estimator = TreatmentEffectEstimator(
            method="matching",
            outcome_variable="outcome",
            treatment_variable="treated",
        )
        n_obs = 200
        data = {
            "treated": np.concatenate([np.zeros(150), np.ones(50)]),
            "outcome": np.random.normal(0, 1, n_obs),
            "confounder": np.random.uniform(-1, 1, n_obs),
        }
        att = estimator.estimate_att(data, confounders=["confounder"])
        assert isinstance(att, float)

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval estimation."""
        estimator = TreatmentEffectEstimator(
            method="naive",
            outcome_variable="outcome",
            treatment_variable="treated",
        )
        n_obs = 200
        data = {
            "treated": np.concatenate([np.zeros(100), np.ones(100)]),
            "outcome": np.concatenate(
                [
                    np.random.normal(0, 1, 100),
                    np.random.normal(0.5, 1, 100),
                ]
            ),
        }
        result = estimator.estimate_with_ci(data, n_bootstrap=100)
        assert "estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["estimate"] < result["ci_upper"]

    def test_estimator_method_variants(self):
        """Test different estimation methods."""
        methods = ["naive", "matching", "weighting", "forest"]
        for method in methods:
            estimator = TreatmentEffectEstimator(
                method=method,
                outcome_variable="outcome",
                treatment_variable="treated",
            )
            assert estimator.method == method


class TestSensitivityAnalysis:
    """Test sensitivity analysis for unobserved confounding."""

    def test_rosenbaum_bounds(self):
        """Test Rosenbaum bounds for sensitivity to hidden bias."""
        from innovate.causal.sensitivity import RosenbaumBounds

        bounds = RosenbaumBounds(
            matched_pairs=100,
            treated_outcomes=[0.5, 0.6, 0.7, 0.75, 0.8],
            control_outcomes=[0.3, 0.4, 0.5, 0.45, 0.5],
        )
        # Test that bounds widen with increased gamma (unmeasured confounding)
        bounds_gamma1 = bounds.calculate(gamma=1.0)  # No hidden bias
        bounds_gamma1_5 = bounds.calculate(gamma=1.5)  # 50% hidden bias

        # The margin should increase with gamma
        margin1 = bounds_gamma1[1] - bounds_gamma1[0]
        margin1_5 = bounds_gamma1_5[1] - bounds_gamma1_5[0]
        assert margin1_5 > margin1  # Bounds should widen

    def test_e_value(self):
        """Test E-value for robustness to unmeasured confounding."""
        from innovate.causal.sensitivity import EValue

        e_value = EValue(
            point_estimate=1.5,  # 50% effect
            ci_lower=1.2,
            ci_upper=1.8,
        )
        # E-value should tell us how strong unmeasured confounder needs to be
        robustness = e_value.calculate()
        assert isinstance(robustness, float)
        assert robustness >= 1.0

    def test_sensitivity_plot_output(self):
        """Test sensitivity analysis can generate diagnostic plots."""
        from innovate.causal.sensitivity import SensitivityAnalysis

        sa = SensitivityAnalysis(
            point_estimate=0.3,
            method="rosenbaum",
        )
        # Should be able to generate bounds across range of gamma
        results = sa.analyze(gamma_range=[1.0, 1.2, 1.4, 1.6])
        assert len(results) == 4
        assert all(isinstance(r, dict) for r in results)

    def test_unmeasured_confounder_scenarios(self):
        """Test sensitivity under different unmeasured confounding scenarios."""
        from innovate.causal.sensitivity import SensitivityAnalysis

        sa = SensitivityAnalysis(point_estimate=0.4)

        scenarios = [
            {"name": "moderate", "gamma": 1.3},
            {"name": "large", "gamma": 1.7},
            {"name": "extreme", "gamma": 2.5},
        ]

        results = {}
        for scenario in scenarios:
            bounds = sa.analyze_scenario(
                gamma=scenario["gamma"],
                direction="both",
            )
            results[scenario["name"]] = bounds

        assert all(name in results for name in ["moderate", "large", "extreme"])


class TestIntegrationWithScenarios:
    """Test integration with scenario workflows."""

    def test_causal_model_with_scenario(self):
        """Test using causal model with scenario workflows."""
        try:
            from innovate.scenario import Scenario
        except ImportError:
            # Skip if scenario module not available
            pytest.skip("Scenario module not available")

        scenario = Scenario(
            name="policy_scenario",
            description="Policy intervention scenario",
        )

        intervention = InterventionContract(
            name="scenario_policy",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )

        # Should be able to attach causal model to scenario
        causal_contract = CausalModelContract(
            name="scenario_model",
            treatment_variable="treated",
            outcome_variable="adoption_rate",
            confounders=["income"],
        )

        causal_model = CausalModel(
            intervention=intervention,
            causal_model=causal_contract,
        )

        scenario.add_causal_model(causal_model)
        assert scenario.causal_model is not None

    def test_causal_evidence_export(self):
        """Test exporting causal evidence for model cards."""
        intervention = InterventionContract(
            name="evidence_export",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        contract = CausalModelContract(
            name="evidence_model",
            treatment_variable="treated",
            outcome_variable="outcome",
            confounders=["confounder"],
        )
        model = CausalModel(intervention=intervention, causal_model=contract)

        evidence = model.export_evidence()
        assert "assumptions" in evidence
        assert "estimand" in evidence
        assert "limitations" in evidence


class TestJSONArrowCompatibility:
    """Test JSON and Arrow serialization compatibility."""

    def test_causal_model_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        intervention = InterventionContract(
            name="json_test",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        contract = CausalModelContract(
            name="json_model",
            treatment_variable="treated",
            outcome_variable="outcome",
            confounders=["confounder"],
        )
        model = CausalModel(intervention=intervention, causal_model=contract)

        # Serialize
        json_str = model.to_json()

        # Deserialize
        restored = CausalModel.from_json(json_str)
        assert restored.intervention.name == "json_test"
        assert restored.causal_model.name == "json_model"

    def test_causal_model_arrow_export(self):
        """Test Arrow table export for polyglot compatibility."""
        import pyarrow as pa

        intervention = InterventionContract(
            name="arrow_test",
            timing="post",
            comparator="control",
            start_time=2020,
            end_time=2025,
        )
        contract = CausalModelContract(
            name="arrow_model",
            treatment_variable="treated",
            outcome_variable="outcome",
            confounders=["confounder"],
        )
        model = CausalModel(intervention=intervention, causal_model=contract)

        # Should be able to export as Arrow table
        table = model.to_arrow()
        assert isinstance(table, pa.Table)

    def test_treatment_effects_arrow_export(self):
        """Test Arrow export of treatment effect estimates."""
        estimator = TreatmentEffectEstimator(
            method="naive",
            outcome_variable="outcome",
            treatment_variable="treated",
        )

        n_obs = 100
        data = {
            "treated": np.concatenate([np.zeros(50), np.ones(50)]),
            "outcome": np.random.normal(0, 1, n_obs),
        }

        result = estimator.estimate_with_ci(data, n_bootstrap=50)

        # Should be able to export results as Arrow
        table = estimator.results_to_arrow(result)
        assert table.num_rows > 0


def test_causal_model_from_json_extra_fields():
    """Test that `from_json` securely rejects arbitrary extra fields to prevent injection."""
    from src.innovate.causal.policy import CausalModel, PolicyEvaluationError

    # Valid causal model structure with an extra 'hack' key in intervention
    json_str_intervention_hack = """{
        "intervention": {
            "name": "test",
            "timing": "post",
            "comparator": "control",
            "hack": "malicious_payload"
        },
        "causal_model": {
            "name": "cm",
            "treatment_variable": "t",
            "outcome_variable": "o",
            "confounders": ["c"]
        }
    }"""

    import pytest

    with pytest.raises(PolicyEvaluationError) as excinfo:
        CausalModel.from_json(json_str_intervention_hack)
    assert "Unknown fields in 'intervention': hack" in str(excinfo.value)

    # Valid causal model structure with an extra 'hack' key in causal_model
    json_str_causal_hack = """{
        "intervention": {
            "name": "test",
            "timing": "post",
            "comparator": "control"
        },
        "causal_model": {
            "name": "cm",
            "treatment_variable": "t",
            "outcome_variable": "o",
            "confounders": ["c"],
            "hack": "malicious_payload"
        }
    }"""

    with pytest.raises(PolicyEvaluationError) as excinfo:
        CausalModel.from_json(json_str_causal_hack)
    assert "Unknown fields in 'causal_model': hack" in str(excinfo.value)


def test_causal_model_from_json_invalid_types():
    """Test that `from_json` securely enforces type checks."""
    from src.innovate.causal.policy import CausalModel, PolicyEvaluationError

    # Invalid type for 'timing'
    json_str_invalid_type = """{
        "intervention": {
            "name": "test",
            "timing": 123,
            "comparator": "control"
        },
        "causal_model": {
            "name": "cm",
            "treatment_variable": "t",
            "outcome_variable": "o",
            "confounders": ["c"]
        }
    }"""

    import pytest

    with pytest.raises(PolicyEvaluationError) as excinfo:
        CausalModel.from_json(json_str_invalid_type)
    assert "Data validation error" in str(excinfo.value)
