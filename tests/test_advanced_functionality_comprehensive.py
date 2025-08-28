"""Comprehensive tests for policy interventions and advanced model functionality."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from innovate.policy.intervention import PolicyIntervention
from innovate.diffuse.bass import BassModel
from innovate.causal.counterfactual import CounterfactualAnalysis
from innovate.ecosystem.complementary_goods import ComplementaryGoodsModel
from innovate.path_dependence.lock_in import LockInModel
from innovate.substitute.fisher_pry import FisherPryModel
from innovate.fitters.scipy_fitter import ScipyFitter


class TestPolicyInterventionComprehensive:
    """Comprehensive tests for policy intervention functionality."""
    
    def setup_method(self):
        """Set up a fitted Bass model for testing."""
        self.bass_model = BassModel()
        self.bass_model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        self.time_points = np.arange(1, 11)
    
    def test_policy_intervention_initialization(self):
        """Test PolicyIntervention initialization."""
        policy = PolicyIntervention(self.bass_model)
        assert policy.model is self.bass_model
        assert policy._original_params == self.bass_model.params_
    
    def test_policy_intervention_unsupported_model(self):
        """Test PolicyIntervention with unsupported model types."""
        from innovate.compete.competition import MultiProductDiffusionModel
        
        multiproduct = MultiProductDiffusionModel(p=[0.02], Q=[[0.1]], m=[1000])
        
        # PolicyIntervention can be initialized with any model, but TypeError is raised during method call
        policy = PolicyIntervention(multiproduct)
        
        with pytest.raises(TypeError, match="currently only supported for BassModel"):
            policy.apply_time_varying_params(t_points=[1, 2, 3])
    
    def test_policy_intervention_no_params(self):
        """Test PolicyIntervention with unfitted model."""
        unfitted_model = BassModel()
        policy = PolicyIntervention(unfitted_model)
        
        with pytest.raises(RuntimeError, match="parameters set before applying policy"):
            policy.apply_time_varying_params(t_points=self.time_points)
    
    def test_apply_time_varying_params_p_effect_only(self):
        """Test applying time-varying effects to p parameter only."""
        policy = PolicyIntervention(self.bass_model)
        
        def p_multiplier(t):
            return 2.0 if t >= 5 else 1.0
        
        predict_func = policy.apply_time_varying_params(
            t_points=self.time_points,
            p_effect=p_multiplier,
            q_effect=None
        )
        
        predictions = predict_func(self.time_points)
        assert len(predictions) == len(self.time_points)
        assert np.all(np.isfinite(predictions))
        assert np.all(predictions >= 0)
        
        # Predictions after t=5 should be different due to p effect
        baseline_predictions = self.bass_model.predict(self.time_points)
        assert not np.allclose(predictions, baseline_predictions)
    
    def test_apply_time_varying_params_q_effect_only(self):
        """Test applying time-varying effects to q parameter only."""
        policy = PolicyIntervention(self.bass_model)
        
        def q_multiplier(t):
            return 1.5 if t >= 3 else 1.0
        
        predict_func = policy.apply_time_varying_params(
            t_points=self.time_points,
            p_effect=None,
            q_effect=q_multiplier
        )
        
        predictions = predict_func(self.time_points)
        assert len(predictions) == len(self.time_points)
        assert np.all(predictions >= 0)
    
    def test_apply_time_varying_params_both_effects(self):
        """Test applying time-varying effects to both p and q parameters."""
        policy = PolicyIntervention(self.bass_model)
        
        def p_multiplier(t):
            return 1.2 if t >= 4 else 1.0
            
        def q_multiplier(t):
            return 0.8 if t >= 6 else 1.0
        
        predict_func = policy.apply_time_varying_params(
            t_points=self.time_points,
            p_effect=p_multiplier,
            q_effect=q_multiplier
        )
        
        predictions = predict_func(self.time_points)
        baseline_predictions = self.bass_model.predict(self.time_points)
        
        # Should be different from baseline due to policy effects
        assert not np.allclose(predictions, baseline_predictions)
    
    def test_policy_extreme_multipliers(self):
        """Test policy intervention with extreme multipliers."""
        policy = PolicyIntervention(self.bass_model)
        
        # Extreme multipliers
        def extreme_p(t):
            return 100.0 if t >= 2 else 1.0
            
        def extreme_q(t):
            return 0.01 if t >= 3 else 1.0
        
        predict_func = policy.apply_time_varying_params(
            t_points=self.time_points,
            p_effect=extreme_p,
            q_effect=extreme_q
        )
        
        predictions = predict_func(self.time_points)
        assert np.all(np.isfinite(predictions))
        # Should not exceed market potential
        assert np.all(predictions <= self.bass_model.params_["m"] + 1e-6)
    
    def test_policy_zero_effects(self):
        """Test policy intervention with zero effects."""
        policy = PolicyIntervention(self.bass_model)
        
        def zero_effect(t):
            return 0.0
        
        predict_func = policy.apply_time_varying_params(
            t_points=self.time_points,
            p_effect=zero_effect,
            q_effect=zero_effect
        )
        
        predictions = predict_func(self.time_points)
        # With zero p and q, should get zero adoption
        assert np.all(predictions == 0)


class TestCounterfactualAnalysis:
    """Test counterfactual analysis functionality."""
    
    def setup_method(self):
        """Set up fitted model for counterfactual analysis."""
        self.model = BassModel()
        self.model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        self.analysis = CounterfactualAnalysis(self.model)
        self.time_horizon = np.arange(1, 11)
    
    def test_counterfactual_initialization(self):
        """Test CounterfactualAnalysis initialization."""
        assert self.analysis.model is self.model
        assert self.analysis.baseline_forecast is None
        assert len(self.analysis.counterfactual_forecasts) == 0
    
    def test_run_baseline(self):
        """Test running baseline forecast."""
        self.analysis.run_baseline(self.time_horizon)
        
        assert self.analysis.baseline_forecast is not None
        assert len(self.analysis.baseline_forecast) == len(self.time_horizon)
        
        # Should match model predictions
        expected = self.model.predict(self.time_horizon)
        np.testing.assert_array_equal(self.analysis.baseline_forecast, expected)
    
    def test_run_counterfactual_scenario(self):
        """Test running counterfactual scenario."""
        self.analysis.run_baseline(self.time_horizon)
        
        counterfactual_params = {"p": 0.04, "q": 0.25}  # Different parameters
        self.analysis.run_counterfactual(
            self.time_horizon,
            counterfactual_params,
            "higher_p_lower_q"
        )
        
        assert "higher_p_lower_q" in self.analysis.counterfactual_forecasts
        forecast = self.analysis.counterfactual_forecasts["higher_p_lower_q"]
        assert len(forecast) == len(self.time_horizon)
    
    def test_run_counterfactual_invalid_parameter(self):
        """Test counterfactual with invalid parameter."""
        self.analysis.run_baseline(self.time_horizon)
        
        invalid_params = {"invalid_param": 0.5}
        
        with pytest.raises(ValueError, match="Parameter 'invalid_param' not found"):
            self.analysis.run_counterfactual(
                self.time_horizon,
                invalid_params,
                "invalid_scenario"
            )
    
    def test_compare_scenarios(self):
        """Test scenario comparison."""
        self.analysis.run_baseline(self.time_horizon)
        
        counterfactual_params = {"m": 1500}  # Larger market potential
        self.analysis.run_counterfactual(
            self.time_horizon,
            counterfactual_params,
            "larger_market"
        )
        
        comparison = self.analysis.compare_scenarios("larger_market")
        
        assert "baseline" in comparison
        assert "counterfactual" in comparison  
        assert "difference" in comparison
        assert "percentage_difference" in comparison
        
        # Larger market should generally lead to higher adoption
        assert np.any(comparison["difference"] > 0)
    
    def test_compare_scenarios_no_baseline(self):
        """Test scenario comparison without baseline."""
        with pytest.raises(RuntimeError, match="Baseline forecast has not been run"):
            self.analysis.compare_scenarios("nonexistent")
    
    def test_compare_scenarios_invalid_scenario(self):
        """Test scenario comparison with invalid scenario name."""
        self.analysis.run_baseline(self.time_horizon)
        
        with pytest.raises(ValueError, match="Counterfactual scenario 'nonexistent' not found"):
            self.analysis.compare_scenarios("nonexistent")


class TestComplementaryGoodsModel:
    """Test complementary goods model functionality."""
    
    def test_complementary_goods_initialization(self):
        """Test ComplementaryGoodsModel initialization."""
        model = ComplementaryGoodsModel()
        
        assert "k1" in model.param_names
        assert "k2" in model.param_names
        assert "c1" in model.param_names  
        assert "c2" in model.param_names
        assert model.params_ == {}
    
    def test_complementary_goods_predict_unfitted(self):
        """Test prediction with unfitted model."""
        model = ComplementaryGoodsModel()
        
        t = [1, 2, 3]
        y0 = np.array([0.1, 0.1])
        
        with pytest.raises(RuntimeError, match="has not been fitted yet"):
            model.predict(t, y0)
    
    def test_complementary_goods_predict_fitted(self):
        """Test prediction with fitted model."""
        model = ComplementaryGoodsModel()
        model.params_ = {"k1": 0.5, "k2": 0.4, "c1": 0.1, "c2": 0.15}
        
        t = [1, 2, 3, 4, 5]
        y0 = np.array([0.1, 0.05])
        
        predictions = model.predict(t, y0)
        
        assert predictions.shape == (len(t), 2)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)  # Should be proportions
    
    def test_complementary_goods_adoption_rate(self):
        """Test adoption rate calculation."""
        model = ComplementaryGoodsModel()
        model.params_ = {"k1": 0.5, "k2": 0.4, "c1": 0.1, "c2": 0.15}
        
        t = [1, 2, 3]
        y0 = np.array([0.1, 0.05])
        
        rates = model.predict_adoption_rate(t, y0)
        
        assert rates.shape == (len(t), 2)
        assert np.all(np.isfinite(rates))
    
    def test_complementary_goods_score_unfitted(self):
        """Test score calculation with unfitted model."""
        model = ComplementaryGoodsModel()
        
        t = [1, 2, 3]
        y = np.array([[0.1, 0.05], [0.2, 0.1], [0.3, 0.15]])
        
        with pytest.raises(RuntimeError, match="has not been fitted yet"):
            model.score(t, y)
    
    def test_complementary_goods_score_fitted(self):
        """Test score calculation with fitted model."""
        model = ComplementaryGoodsModel()
        model.params_ = {"k1": 0.5, "k2": 0.4, "c1": 0.1, "c2": 0.15}
        
        t = [1, 2, 3]
        y0 = np.array([0.01, 0.01])
        y_pred = model.predict(t, y0)
        
        # Perfect predictions should give R² ≈ 1
        score = model.score(t, y_pred)
        assert 0.95 <= score <= 1.0


class TestLockInModel:
    """Test path dependence lock-in model functionality."""
    
    def test_lock_in_initialization(self):
        """Test LockInModel initialization."""
        model = LockInModel()
        
        expected_params = ["alpha", "beta", "gamma", "delta", "m"]
        assert model.param_names == expected_params
        assert model.params_ == {}
    
    def test_lock_in_differential_equation(self):
        """Test differential equation calculation."""
        model = LockInModel()
        
        y = [0.3, 0.2]  # Current adoption levels
        t = 5.0
        params = [0.5, 0.3, 0.2, 0.4, 1.0]  # alpha, beta, gamma, delta, m
        
        dydt = model.differential_equation(y, t, *params)
        
        assert len(dydt) == 2
        assert np.all(np.isfinite(dydt))
    
    def test_lock_in_predict_unfitted(self):
        """Test prediction with unfitted model."""
        model = LockInModel()
        
        t = [1, 2, 3]
        y0 = [0.1, 0.05]
        
        with pytest.raises(RuntimeError, match="has not been fitted yet"):
            model.predict(t, y0)
    
    def test_lock_in_predict_fitted(self):
        """Test prediction with fitted model."""
        model = LockInModel()
        model.params_ = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "delta": 0.4, "m": 1.0}
        
        t = np.linspace(0, 10, 11)
        y0 = [0.1, 0.05]
        
        predictions = model.predict(t, y0)
        
        assert predictions.shape == (len(t), 2)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= model.params_["m"])
    
    def test_lock_in_fit_invalid_data_shape(self):
        """Test fitting with invalid data shape."""
        model = LockInModel()
        
        t = [1, 2, 3]
        y_1d = [0.1, 0.2, 0.3]  # 1D array instead of 2D
        
        with pytest.raises(ValueError, match="2D array with two columns"):
            model.fit(t, y_1d)
    
    def test_lock_in_bounds_and_initial_guesses(self):
        """Test bounds and initial guesses."""
        model = LockInModel()
        
        t = [1, 2, 3, 4, 5]
        y = np.array([[0.1, 0.05], [0.15, 0.08], [0.2, 0.12], [0.25, 0.16], [0.3, 0.2]])
        
        bounds = model.bounds(t, y)
        initial_guesses = model.initial_guesses(t, y)
        
        assert len(bounds) == len(model.param_names)
        assert len(initial_guesses) == len(model.param_names)
        
        # Check bounds are reasonable
        for param_name, (lower, upper) in bounds.items():
            assert lower < upper
            assert initial_guesses[param_name] >= lower
            assert initial_guesses[param_name] <= upper


class TestFisherPryModelExtended:
    """Extended tests for Fisher-Pry substitution model."""
    
    def test_fisher_pry_edge_cases(self):
        """Test Fisher-Pry model with edge cases."""
        model = FisherPryModel()
        
        # Test with all zeros
        t = [1, 2, 3, 4, 5] 
        y_zeros = [0, 0, 0, 0, 0]
        
        initial_guesses = model.initial_guesses(t, y_zeros)
        assert np.all(np.isfinite(list(initial_guesses.values())))
        
        # Test with all ones (complete substitution)
        y_ones = [1, 1, 1, 1, 1]
        initial_guesses_ones = model.initial_guesses(t, y_ones)
        assert np.all(np.isfinite(list(initial_guesses_ones.values())))
        
        # Test with values outside [0,1] range
        y_invalid = [-0.1, 0.5, 1.2, 0.8, 0.3]
        
        # Should handle gracefully or raise appropriate error
        try:
            bounds = model.bounds(t, y_invalid)
            assert len(bounds) == 2  # alpha and t0
        except ValueError:
            # Invalid range handling is acceptable
            pass
    
    def test_fisher_pry_differential_equation(self):
        """Test Fisher-Pry differential equation."""
        model = FisherPryModel()
        
        y = 0.3  # 30% market share
        t = 5.0
        alpha = 0.2
        t0 = 10.0
        
        dydt = model.differential_equation(y, t, alpha=alpha, t0=t0)
        
        assert np.isfinite(dydt)
        # Should be positive when t < t0 and y < 0.5 (still growing)
        if t < t0 and y < 0.5:
            assert dydt > 0
    
    def test_fisher_pry_predict_adoption_rate_unfitted(self):
        """Test adoption rate prediction with unfitted model."""
        model = FisherPryModel()
        
        t = [1, 2, 3]
        
        with pytest.raises(RuntimeError, match="has not been fitted yet"):
            model.predict_adoption_rate(t)
    
    def test_fisher_pry_predict_adoption_rate_fitted(self):
        """Test adoption rate prediction with fitted model.""" 
        model = FisherPryModel()
        model.params_ = {"alpha": 0.2, "t0": 10.0}
        
        t = [5, 10, 15, 20]
        rates = model.predict_adoption_rate(t)
        
        assert len(rates) == len(t)
        assert np.all(np.isfinite(rates))
    
    def test_fisher_pry_saturation_behavior(self):
        """Test Fisher-Pry model saturation behavior."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 10.0}
        
        # Very large time values should approach 1.0 (complete substitution)
        t_large = [50, 100, 1000]
        predictions = model.predict(t_large)
        
        assert np.all(predictions >= 0.99)  # Should be very close to 1


class TestAdvancedModelInteractions:
    """Test interactions between different advanced models."""
    
    def test_model_composition_workflow(self):
        """Test a complete workflow using multiple advanced models."""
        # Create base diffusion
        bass = BassModel()
        bass.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Apply policy intervention
        policy = PolicyIntervention(bass)
        def policy_effect(t):
            return 1.5 if t >= 5 else 1.0
        
        policy_predictor = policy.apply_time_varying_params(
            t_points=np.arange(1, 11),
            p_effect=policy_effect
        )
        
        # Run counterfactual analysis
        counterfactual = CounterfactualAnalysis(bass)
        counterfactual.run_baseline(np.arange(1, 11))
        counterfactual.run_counterfactual(
            np.arange(1, 11),
            {"p": 0.03},  # Higher innovation parameter
            "higher_innovation"
        )
        
        comparison = counterfactual.compare_scenarios("higher_innovation")
        
        # Verify the workflow produces sensible results
        assert len(comparison["baseline"]) == 10
        assert len(comparison["counterfactual"]) == 10
        assert np.any(comparison["difference"] != 0)  # Should be some difference
    
    def test_parameter_sensitivity_analysis(self):
        """Test parameter sensitivity across different models."""
        models = [
            (BassModel(), {"p": 0.02, "q": 0.3, "m": 1000}),
            (FisherPryModel(), {"alpha": 0.2, "t0": 10.0})
        ]
        
        t = np.arange(1, 21)
        sensitivity_results = {}
        
        for model, base_params in models:
            model.params_ = base_params
            baseline = model.predict(t)
            
            # Test sensitivity to each parameter
            for param_name, base_value in base_params.items():
                # 10% increase
                modified_params = base_params.copy()
                modified_params[param_name] = base_value * 1.1
                
                model.params_ = modified_params
                modified_prediction = model.predict(t)
                
                # Calculate sensitivity (percentage change in output / percentage change in input)
                relative_change = np.mean(np.abs(modified_prediction - baseline) / baseline)
                sensitivity_results[f"{model.__class__.__name__}_{param_name}"] = relative_change
                
                # Restore original parameters
                model.params_ = base_params
        
        # Verify sensitivity analysis produces reasonable results
        assert all(v >= 0 for v in sensitivity_results.values())
        assert all(np.isfinite(v) for v in sensitivity_results.values())
    
    def test_error_propagation(self):
        """Test error propagation through model chains."""
        # Create a chain: Base model -> Policy -> Counterfactual
        bass = BassModel()
        bass.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Introduce error in base parameters
        noisy_params = {
            "p": bass.params_["p"] * (1 + np.random.normal(0, 0.1)),
            "q": bass.params_["q"] * (1 + np.random.normal(0, 0.1)), 
            "m": bass.params_["m"] * (1 + np.random.normal(0, 0.1))
        }
        
        # Ensure parameters remain positive
        noisy_params = {k: max(v, 1e-6) for k, v in noisy_params.items()}
        
        noisy_bass = BassModel()
        noisy_bass.params_ = noisy_params
        
        t = np.arange(1, 11)
        
        # Compare clean vs noisy predictions
        clean_pred = bass.predict(t)
        noisy_pred = noisy_bass.predict(t)
        
        # Error should be bounded
        relative_error = np.abs(noisy_pred - clean_pred) / clean_pred
        assert np.all(relative_error < 1.0)  # Error shouldn't exceed 100%
    
    def test_model_validation_pipeline(self):
        """Test a complete model validation pipeline."""
        # Generate synthetic data
        true_model = BassModel()
        true_model.params_ = {"p": 0.025, "q": 0.35, "m": 1200}
        
        t_train = np.arange(1, 21)
        t_test = np.arange(21, 31)
        
        y_train_clean = true_model.predict(t_train)
        y_test_clean = true_model.predict(t_test)
        
        # Add noise
        np.random.seed(42)
        y_train = y_train_clean + np.random.normal(0, 0.05 * y_train_clean)
        y_test = y_test_clean + np.random.normal(0, 0.05 * y_test_clean)
        
        # Fit model
        fitted_model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(fitted_model, t_train, y_train)
        
        # Validate on test set
        test_predictions = fitted_model.predict(t_test)
        test_rmse = np.sqrt(np.mean((y_test - test_predictions)**2))
        
        # Performance should be reasonable
        assert test_rmse < np.std(y_test)  # Better than just predicting mean
        assert fitted_model.params_ is not None
        assert all(v > 0 for v in fitted_model.params_.values())