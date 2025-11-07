"""Comprehensive tests for MixtureModel to improve coverage to >90%."""
import numpy as np
import pytest
from innovate.models.mixture import MixtureModel
from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel


class TestMixtureModelComprehensive:
    """Comprehensive tests for MixtureModel to ensure >90% coverage."""
    
    def test_mixture_model_init_edge_cases(self):
        """Test initialization edge cases."""
        # Test with empty models list
        with pytest.raises(ValueError, match="At least one model is required."):
            MixtureModel(models=[], weights=[])
        
        # Test with inconsistent weights length
        bass_model = BassModel()
        with pytest.raises(ValueError, match="Number of weights must match number of models."):
            MixtureModel(models=[bass_model], weights=[0.5, 0.6])
        
        # Test with weights that don't sum to 1 (use 2 models to match 2 weights)
        bass_model2 = BassModel()
        with pytest.raises(ValueError, match="Weights must sum to 1."):
            MixtureModel(models=[bass_model, bass_model2], weights=[0.5, 0.6])
        
        # Test initialization with single model
        model = MixtureModel(models=[bass_model])
        assert model.num_components == 1
        assert len(model.weights) == 1
        assert model.weights[0] == 1.0  # Default weight for single component
    
    def test_mixture_model_param_names(self):
        """Test the param_names property."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        param_names = model.param_names
        
        # Check that parameters from both models are included
        # Bass params should have "model_0_" prefix
        assert "model_0_p" in param_names
        assert "model_0_q" in param_names
        assert "model_0_m" in param_names
        # Logistic params should have "model_1_" prefix
        assert "model_1_L" in param_names
        assert "model_1_k" in param_names
        assert "model_1_x0" in param_names
        # Weight parameters should be included
        assert "weight_0" in param_names
        assert "weight_1" in param_names
    
    def test_initial_guesses_method(self):
        """Test the initial_guesses method."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        t = [0, 1, 2, 3, 4]
        y = [10, 20, 30, 40, 50]
        
        guesses = model.initial_guesses(t, y)
        
        # Check that parameters from both models are included
        assert "model_0_p" in guesses
        assert "model_0_q" in guesses
        assert "model_0_m" in guesses
        assert "model_1_L" in guesses
        assert "model_1_k" in guesses
        assert "model_1_x0" in guesses
        # Weight parameters with default values
        assert "weight_0" in guesses
        assert "weight_1" in guesses
        assert guesses["weight_0"] == 0.5  # 1 / num_components
        assert guesses["weight_1"] == 0.5  # 1 / num_components
    
    def test_bounds_method(self):
        """Test the bounds method."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        t = [0, 1, 2, 3, 4]
        y = [10, 20, 30, 40, 50]
        
        bounds = model.bounds(t, y)
        
        # Check that parameters from both models are included
        assert "model_0_p" in bounds
        assert "model_0_q" in bounds
        assert "model_0_m" in bounds
        assert "model_1_L" in bounds
        assert "model_1_k" in bounds
        assert "model_1_x0" in bounds
        # Weight parameters with (0, 1) bounds
        assert "weight_0" in bounds
        assert "weight_1" in bounds
        assert bounds["weight_0"] == (0, 1)
        assert bounds["weight_1"] == (0, 1)
    
    def test_predict_unfitted_model_error(self):
        """Test that predict raises error when model is not fitted."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict([0, 1, 2])
    
    def test_score_method(self):
        """Test the score method."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        # Set some parameters manually to avoid needing to fit
        model._params = {
            "model_0_p": 0.03, "model_0_q": 0.38, "model_0_m": 1000,
            "model_1_L": 1000, "model_1_k": 0.2, "model_1_x0": 10,
            "weight_0": 0.5, "weight_1": 0.5
        }
        
        # Set the internal model parameters too
        bass_model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        logistic_model.params_ = {"L": 1000, "k": 0.2, "x0": 10}
        
        t = [0, 1, 2, 3, 4]
        y = [10, 20, 30, 40, 50]
        
        score = model.score(t, y)
        
        assert isinstance(score, float)
        # Score can be negative for poor fits but should be finite
    
    def test_predict_adoption_rate_unfitted_model_error(self):
        """Test that predict_adoption_rate raises error when model is not fitted."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict_adoption_rate([0, 1, 2])
    
    def test_predict_adoption_rate_method(self):
        """Test the predict_adoption_rate method."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        # Set some parameters manually to avoid needing to fit
        model._params = {
            "model_0_p": 0.03, "model_0_q": 0.38, "model_0_m": 1000,
            "model_1_L": 1000, "model_1_k": 0.2, "model_1_x0": 10,
            "weight_0": 0.5, "weight_1": 0.5
        }
        
        # Set the internal model parameters too
        bass_model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        logistic_model.params_ = {"L": 1000, "k": 0.2, "x0": 10}
        
        t = [0, 1, 2, 3, 4]
        
        rates = model.predict_adoption_rate(t)
        
        assert isinstance(rates, np.ndarray)
        assert len(rates) == len(t)
    
    def test_property_setters_and_getters(self):
        """Test the params_ property setter and getter."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        # Test getter with empty params
        assert model.params_ == {}
        
        # Test setter and getter
        params = {
            "model_0_p": 0.03, "model_0_q": 0.38, "model_0_m": 1000,
            "model_1_L": 1000, "model_1_k": 0.2, "model_1_x0": 10,
            "weight_0": 0.4, "weight_1": 0.6
        }
        model.params_ = params
        assert model.params_ == params
        
        # Check that internal models were updated too
        assert bass_model.params_ == {"p": 0.03, "q": 0.38, "m": 1000}
        assert logistic_model.params_ == {"L": 1000, "k": 0.2, "x0": 10}
        
        # Check that weights were updated
        assert model.weights[0] == 0.4
        assert model.weights[1] == 0.6
    
    def test_update_params_from_models(self):
        """Test the _update_params_from_models method."""
        bass_model = BassModel()
        bass_model.params_ = {"p": 0.05, "q": 0.4, "m": 1500}
        logistic_model = LogisticModel()
        logistic_model.params_ = {"L": 1500, "k": 0.3, "x0": 15}
        
        model = MixtureModel(models=[bass_model, logistic_model])
        model.weights = np.array([0.3, 0.7])
        
        # Manually call the update method
        model._update_params_from_models()
        
        # Check that main params were updated
        expected_params = {
            "model_0_p": 0.05, "model_0_q": 0.4, "model_0_m": 1500,
            "model_1_L": 1500, "model_1_k": 0.3, "model_1_x0": 15,
            "weight_0": 0.3, "weight_1": 0.7
        }
        assert model.params_ == expected_params
    
    def test_predict_with_covariates(self):
        """Test predict method with covariates (should pass them to submodels)."""
        bass_model = BassModel(covariates=["advertising"])
        logistic_model = LogisticModel(covariates=["advertising"])
        model = MixtureModel(models=[bass_model, logistic_model])
        
        # Set parameters
        model._params = {
            "model_0_p": 0.03, "model_0_q": 0.38, "model_0_m": 1000,
            "model_1_L": 1000, "model_1_k": 0.2, "model_1_x0": 10,
            "weight_0": 0.5, "weight_1": 0.5
        }
        
        bass_model.params_ = {"p": 0.03, "q": 0.38, "m": 1000, 
                             "beta_p_advertising": 0.001, "beta_q_advertising": 0.002, "beta_m_advertising": 1.0}
        logistic_model.params_ = {"L": 1000, "k": 0.2, "x0": 10,
                                 "beta_L_advertising": 1.0, "beta_k_advertising": 0.002, "beta_x0_advertising": 0.1}
        
        t = [0, 1, 2]
        covariates = {"advertising": [0.1, 0.2, 0.3]}
        
        # Should work without error
        predictions = model.predict(t, covariates)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(t)
    
    def test_fit_method_basic(self):
        """Test the basic functionality of the fit method."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model], max_iter=2)  # Reduce iterations for faster test
        
        t = [0, 1, 2, 3, 4]
        y = [10, 20, 30, 40, 50]
        
        # This will run the EM algorithm
        fitted_model = model.fit(t, y)
        assert fitted_model is not None
        assert fitted_model._params  # Should have some parameters set after fitting
    
    def test_differential_equation_method(self):
        """Test the differential_equation method (currently just a pass)."""
        bass_model = BassModel()
        logistic_model = LogisticModel()
        model = MixtureModel(models=[bass_model, logistic_model])
        
        # This method literally just has a "pass" in it, so calling it should not error
        try:
            model.differential_equation(None, None, None)
            # If no error, that means the method exists and doesn't crash
            assert True
        except Exception:
            assert False  # Should not raise an exception


def test_mixture_model_comprehensive_integration():
    """Integration test for all MixtureModel functionality."""
    # Create model with multiple sub-models
    bass_model = BassModel()
    logistic_model = LogisticModel()
    model = MixtureModel(models=[bass_model, logistic_model], weights=[0.6, 0.4])
    
    # Check param names include all expected parameters
    param_names = model.param_names
    
    # Parameters from bass model should be prefixed with model_0_
    assert "model_0_p" in param_names
    assert "model_0_q" in param_names
    assert "model_0_m" in param_names
    
    # Parameters from logistic model should be prefixed with model_1_
    assert "model_1_L" in param_names
    assert "model_1_k" in param_names
    assert "model_1_x0" in param_names
    
    # Weight parameters should be included
    assert "weight_0" in param_names
    assert "weight_1" in param_names
    
    # Test initial guesses
    t = [0, 1, 2, 3, 4]
    y = [10, 20, 30, 40, 50]
    guesses = model.initial_guesses(t, y)
    
    for name in param_names:
        assert name in guesses
    
    # Test bounds
    bounds = model.bounds(t, y)
    
    for name in param_names:
        assert name in bounds
        if name.startswith("weight_"):
            # Weight bounds should be (0, 1)
            assert bounds[name] == (0, 1)
        else:
            # Other bounds come from submodels
            assert isinstance(bounds[name], tuple)
            assert len(bounds[name]) == 2


if __name__ == "__main__":
    # Run the tests individually to ensure they work
    test_instance = TestMixtureModelComprehensive()
    
    print("Running MixtureModel comprehensive tests...")
    
    test_instance.test_mixture_model_init_edge_cases()
    print("✓ Mixture model init edge cases test passed")
    
    test_instance.test_mixture_model_param_names()
    print("✓ Mixture model param names test passed")
    
    test_instance.test_initial_guesses_method()
    print("✓ Initial guesses method test passed")
    
    test_instance.test_bounds_method()
    print("✓ Bounds method test passed")
    
    test_instance.test_predict_unfitted_model_error()
    print("✓ Predict unfitted model error test passed")
    
    test_instance.test_score_method()
    print("✓ Score method test passed")
    
    test_instance.test_predict_adoption_rate_unfitted_model_error()
    print("✓ Predict adoption rate unfitted model error test passed")
    
    test_instance.test_predict_adoption_rate_method()
    print("✓ Predict adoption rate method test passed")
    
    test_instance.test_property_setters_and_getters()
    print("✓ Property setters and getters test passed")
    
    test_instance.test_update_params_from_models()
    print("✓ Update params from models test passed")
    
    test_instance.test_predict_with_covariates()
    print("✓ Predict with covariates test passed")
    
    test_instance.test_fit_method_basic()
    print("✓ Fit method basic test passed")
    
    test_instance.test_differential_equation_method()
    print("✓ Differential equation method test passed")
    
    # Run integration test
    test_mixture_model_comprehensive_integration()
    print("✓ Integration test passed")
    
    print("\nAll comprehensive MixtureModel tests passed! 🎉")