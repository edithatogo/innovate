"""Tests for the counterfactual analysis module."""

import pytest

from src.innovate.causal.counterfactual import CounterfactualAnalysis
from src.innovate.diffuse.bass import BassModel


class TestCounterfactualAnalysis:
    """Test cases for CounterfactualAnalysis class."""

    def test_init_with_unfitted_model_raises_error(self):
        """Test that initializing with an unfitted model raises an error."""
        # Create an unfitted model
        model = BassModel()

        with pytest.raises(ValueError, match="The model must be fitted before"):
            CounterfactualAnalysis(model)

    def test_init_with_fitted_model(self):
        """Test that initializing with a fitted model works."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)
        assert analysis.model == model
        assert analysis.baseline_forecast is None
        assert analysis.counterfactual_forecasts == {}

    def test_run_baseline(self):
        """Test running baseline forecast."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)
        t = [1, 2, 3, 4, 5]

        analysis.run_baseline(t)
        assert analysis.baseline_forecast is not None
        assert len(analysis.baseline_forecast) == len(t)

    def test_run_counterfactual(self):
        """Test running counterfactual scenario."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)
        t = [1, 2, 3, 4, 5]

        # Run baseline first
        analysis.run_baseline(t)

        # Run counterfactual with modified parameters
        analysis.run_counterfactual(
            scenario_name="higher_p",
            t=t,
            counterfactual_params={"p": 0.05},  # Higher innovation coefficient
        )

        assert "higher_p" in analysis.counterfactual_forecasts
        assert len(analysis.counterfactual_forecasts["higher_p"]) == len(t)

    def test_run_counterfactual_with_invalid_param(self):
        """Test running counterfactual with invalid parameter raises error."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)
        t = [1, 2, 3, 4, 5]

        with pytest.raises(ValueError, match="Parameter 'invalid_param' not found"):
            analysis.run_counterfactual(scenario_name="invalid", t=t, counterfactual_params={"invalid_param": 0.05})

    def test_compare_scenarios_before_baseline_raises_error(self):
        """Test that comparing scenarios without running baseline raises error."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)

        with pytest.raises(RuntimeError, match="Baseline forecast has not been run"):
            analysis.compare_scenarios("nonexistent")

    def test_compare_scenarios_nonexistent_scenario(self):
        """Test that comparing non-existent scenario raises error."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)
        t = [1, 2, 3, 4, 5]

        # Run baseline
        analysis.run_baseline(t)

        with pytest.raises(ValueError, match="Counterfactual scenario 'nonexistent' not found"):
            analysis.compare_scenarios("nonexistent")

    def test_compare_scenarios_with_valid_scenario(self):
        """Test comparing a valid counterfactual scenario."""
        # Create a fitted model
        model = BassModel()
        model.params_ = {"p": 0.01, "q": 0.15, "m": 1000}

        analysis = CounterfactualAnalysis(model)
        t = [1, 2, 3, 4, 5]

        # Run baseline
        analysis.run_baseline(t)

        # Run counterfactual
        analysis.run_counterfactual(scenario_name="test_scenario", t=t, counterfactual_params={"p": 0.05})

        # Compare scenarios
        results = analysis.compare_scenarios("test_scenario")

        assert "baseline" in results
        assert "counterfactual" in results
        assert "difference" in results
        assert "percentage_difference" in results

        assert len(results["baseline"]) == len(t)
        assert len(results["counterfactual"]) == len(t)
        assert len(results["difference"]) == len(t)
        assert len(results["percentage_difference"]) == len(t)

    def test_compare_scenarios_with_zero_baseline_values(self):
        """Test comparing scenarios when baseline has zeros (to test division by zero)."""
        # Create a fitted model that might produce zeros
        model = BassModel()
        model.params_ = {"p": 0.0, "q": 0.0, "m": 1000}  # This should produce a flat forecast

        analysis = CounterfactualAnalysis(model)
        t = [1, 2, 3, 4, 5]

        # Run baseline
        analysis.run_baseline(t)

        # Run counterfactual with different parameters
        analysis.run_counterfactual(scenario_name="test_scenario", t=t, counterfactual_params={"p": 0.05})

        # Compare scenarios - this should handle division by zero scenarios
        results = analysis.compare_scenarios("test_scenario")

        assert "baseline" in results
        assert "counterfactual" in results
        assert "difference" in results
        assert "percentage_difference" in results
