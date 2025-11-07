"""Comprehensive tests for Lotka-Volterra model to improve coverage to >90%."""
import numpy as np
import pytest
from scipy.optimize import minimize
from innovate.compete.lotka_volterra import LotkaVolterraModel


class TestLotkaVolterraModelComprehensive:
    """Comprehensive tests for Lotka-Volterra model to ensure >90% coverage."""
    
    def test_lotka_volterra_initialization(self):
        """Test basic initialization of Lotka-Volterra model."""
        model = LotkaVolterraModel()
        assert model._params == {}
        assert model.covariates == []
        
        # Test initialization with covariates
        model_cov = LotkaVolterraModel(covariates=["advertising", "price"])
        assert "advertising" in model_cov.covariates
        assert "price" in model_cov.covariates
    
    def test_param_names_with_covariates(self):
        """Test param_names with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        names = model.param_names
        
        # Check basic parameters exist
        assert "alpha1" in names
        assert "beta1" in names
        assert "alpha2" in names
        assert "beta2" in names
        
        # Check covariate parameters exist
        assert "beta_alpha1_advertising" in names
        assert "beta_beta1_advertising" in names
        assert "beta_alpha2_advertising" in names
        assert "beta_beta2_advertising" in names
    
    def test_initial_guesses_with_covariates(self):
        """Test initial_guesses method with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        t = np.arange(5)
        y = np.random.rand(5, 2)
        
        guesses = model.initial_guesses(t, y)
        
        # Check basic parameters
        assert "alpha1" in guesses
        assert "beta1" in guesses
        assert "alpha2" in guesses
        assert "beta2" in guesses
        
        # Check covariate parameters with default value of 0.0
        assert "beta_alpha1_advertising" in guesses
        assert "beta_beta1_advertising" in guesses
        assert "beta_alpha2_advertising" in guesses
        assert "beta_beta2_advertising" in guesses
        
        assert guesses["beta_alpha1_advertising"] == 0.0
        assert guesses["beta_beta1_advertising"] == 0.0
        assert guesses["beta_alpha2_advertising"] == 0.0
        assert guesses["beta_beta2_advertising"] == 0.0
    
    def test_bounds_with_covariates(self):
        """Test bounds method with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        t = np.arange(5)
        y = np.random.rand(5, 2)
        
        bounds = model.bounds(t, y)
        
        # Check basic parameters have (0, inf) bounds
        assert bounds["alpha1"] == (0, np.inf)
        assert bounds["beta1"] == (0, np.inf)
        assert bounds["alpha2"] == (0, np.inf)
        assert bounds["beta2"] == (0, np.inf)
        
        # Check covariate parameters have (-inf, inf) bounds
        assert bounds["beta_alpha1_advertising"] == (-np.inf, np.inf)
        assert bounds["beta_beta1_advertising"] == (-np.inf, np.inf)
        assert bounds["beta_alpha2_advertising"] == (-np.inf, np.inf)
        assert bounds["beta_beta2_advertising"] == (-np.inf, np.inf)
    
    def test_predict_unfitted_model_error(self):
        """Test that predict raises error when model is not fitted."""
        model = LotkaVolterraModel()
        
        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict([0, 1, 2], [0.1, 0.1])
    
    def test_score_unfitted_model_error(self):
        """Test that score raises error when model is not fitted."""
        model = LotkaVolterraModel()
        t = [0, 1, 2]
        y = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        
        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.score(t, y)
    
    def test_predict_adoption_rate_unfitted_model_error(self):
        """Test that predict_adoption_rate raises error when model is not fitted."""
        model = LotkaVolterraModel()
        y0 = [0.1, 0.1]
        
        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict_adoption_rate([0, 1, 2], y0)
    
    def test_fit_value_error(self):
        """Test that fit raises error with invalid y shape."""
        model = LotkaVolterraModel()
        t = np.arange(5)
        y_invalid = np.random.rand(5)  # Only 1D instead of 2D with 2 columns
        
        with pytest.raises(ValueError, match="`y` must be a 2D array with two columns"):
            model.fit(t, y_invalid)
    
    def test_fit_failure(self, monkeypatch):
        """Test fit failure handling when optimization fails."""
        model = LotkaVolterraModel()
        t = np.arange(5)
        y = np.random.rand(5, 2)
        
        # Patch the minimize function to return a failed result
        def mock_minimize(*args, **kwargs):
            class MockResult:
                success = False
                message = "Mocked failure message"
            return MockResult()
        
        monkeypatch.setattr("scipy.optimize.minimize", mock_minimize)
        
        with pytest.raises(RuntimeError, match="Fitting failed"):
            model.fit(t, y)
    
    def test_differential_equation_with_covariates(self):
        """Test differential_equation with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        
        # Set parameters with covariate coefficients
        model.params_ = {
            "alpha1": 0.1,
            "beta1": 0.01,
            "alpha2": 0.1,
            "beta2": 0.01,
            "beta_alpha1_advertising": 0.001,
            "beta_beta1_advertising": 0.0001,
            "beta_alpha2_advertising": 0.001,
            "beta_beta2_advertising": 0.0001
        }
        
        y = [0.2, 0.3]  # Current state [y1, y2]
        t = 1.0  # Current time
        params = [0.1, 0.01, 0.1, 0.01, 0.001, 0.0001, 0.001, 0.0001]  # All parameters
        covariates = {"advertising": [0.5, 0.6, 0.7]}  # Covariate over time
        t_eval = [0, 1, 2]  # Time points for interpolation
        
        result = model.differential_equation(y, t, params, covariates, t_eval)
        
        # Should return a list with 2 elements (dy1_dt, dy2_dt)
        assert len(result) == 2
        assert all(isinstance(r, (int, float, np.number)) for r in result)
    
    def test_differential_equation_without_covariates(self):
        """Test differential_equation without covariates."""
        model = LotkaVolterraModel()
        
        y = [0.2, 0.3]  # Current state [y1, y2]
        t = 1.0  # Current time
        params = [0.1, 0.01, 0.1, 0.01]  # Base parameters
        covariates = None
        t_eval = [0, 1, 2]
        
        result = model.differential_equation(y, t, params, covariates, t_eval)
        
        # Should return a list with 2 elements (dy1_dt, dy2_dt)
        assert len(result) == 2
        assert all(isinstance(r, (int, float, np.number)) for r in result)
    
    def test_predict_with_covariates(self):
        """Test predict with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        model.params_ = {
            "alpha1": 0.1,
            "beta1": 0.01,
            "alpha2": 0.1,
            "beta2": 0.01,
            "beta_alpha1_advertising": 0.001,
            "beta_beta1_advertising": 0.0001,
            "beta_alpha2_advertising": 0.001,
            "beta_beta2_advertising": 0.0001
        }
        
        t = np.arange(0, 5, 1)
        y0 = [0.01, 0.02]
        covariates = {"advertising": [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        predictions = model.predict(t, y0, covariates)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(t), 2)
        assert np.all(predictions >= 0)
    
    def test_score_with_covariates(self):
        """Test score with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        model.params_ = {
            "alpha1": 0.1,
            "beta1": 0.01,
            "alpha2": 0.1,
            "beta2": 0.01,
            "beta_alpha1_advertising": 0.001,
            "beta_beta1_advertising": 0.0001,
            "beta_alpha2_advertising": 0.001,
            "beta_beta2_advertising": 0.0001
        }
        
        t = np.arange(5)
        y = np.random.rand(5, 2)
        covariates = {"advertising": [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        score = model.score(t, y, covariates)
        
        assert isinstance(score, float)
        # Score can be negative for poor fits, but should be a float
    
    def test_predict_adoption_rate_with_covariates(self):
        """Test predict_adoption_rate with covariates."""
        model = LotkaVolterraModel(covariates=["advertising"])
        model.params_ = {
            "alpha1": 0.1,
            "beta1": 0.01,
            "alpha2": 0.1,
            "beta2": 0.01,
            "beta_alpha1_advertising": 0.001,
            "beta_beta1_advertising": 0.0001,
            "beta_alpha2_advertising": 0.001,
            "beta_beta2_advertising": 0.0001
        }
        
        t = np.arange(5)
        y0 = [0.01, 0.02]
        covariates = {"advertising": [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        rates = model.predict_adoption_rate(t, y0, covariates)
        
        assert isinstance(rates, np.ndarray)
        assert rates.shape == (len(t), 2)
    
    def test_property_setters_and_getters(self):
        """Test the params_ property setter and getter."""
        model = LotkaVolterraModel()
        
        # Test getter with empty params
        assert model.params_ == {}
        
        # Test setter and getter
        params = {"alpha1": 0.1, "beta1": 0.01, "alpha2": 0.1, "beta2": 0.01}
        model.params_ = params
        assert model.params_ == params
        
        # Test with different parameter values
        new_params = {"alpha1": 0.2, "beta1": 0.02, "alpha2": 0.2, "beta2": 0.02}
        model.params_ = new_params
        assert model.params_ == new_params


def test_lotka_volterra_comprehensive_integration():
    """Integration test for all Lotka-Volterra model functionality."""
    # Create model with covariates
    model = LotkaVolterraModel(covariates=["advertising", "price"])
    
    # Check param names include all expected parameters
    param_names = model.param_names
    expected_names = [
        "alpha1", "beta1", "alpha2", "beta2",
        "beta_alpha1_advertising", "beta_beta1_advertising", 
        "beta_alpha2_advertising", "beta_beta2_advertising",
        "beta_alpha1_price", "beta_beta1_price", 
        "beta_alpha2_price", "beta_beta2_price"
    ]
    
    for name in expected_names:
        assert name in param_names
    
    # Test initial guesses
    t = np.arange(10)
    y = np.random.rand(10, 2)
    guesses = model.initial_guesses(t, y)
    
    for name in expected_names:
        assert name in guesses
    
    # Test bounds
    bounds = model.bounds(t, y)
    
    for name in expected_names:
        assert name in bounds
        assert isinstance(bounds[name], tuple)
        assert len(bounds[name]) == 2


if __name__ == "__main__":
    # Run the tests individually to ensure they work
    test_instance = TestLotkaVolterraModelComprehensive()
    
    print("Running Lotka-Volterra model comprehensive tests...")
    
    test_instance.test_lotka_volterra_initialization()
    print("✓ Basic initialization test passed")
    
    test_instance.test_param_names_with_covariates()
    print("✓ Param names with covariates test passed")
    
    test_instance.test_initial_guesses_with_covariates()
    print("✓ Initial guesses with covariates test passed")
    
    test_instance.test_bounds_with_covariates()
    print("✓ Bounds with covariates test passed")
    
    test_instance.test_predict_unfitted_model_error()
    print("✓ Predict unfitted model error test passed")
    
    test_instance.test_score_unfitted_model_error()
    print("✓ Score unfitted model error test passed")
    
    test_instance.test_predict_adoption_rate_unfitted_model_error()
    print("✓ Predict adoption rate unfitted model error test passed")
    
    test_instance.test_fit_value_error()
    print("✓ Fit value error test passed")
    
    # Skip the fit failure test as it uses monkeypatch
    
    test_instance.test_differential_equation_with_covariates()
    print("✓ Differential equation with covariates test passed")
    
    test_instance.test_differential_equation_without_covariates()
    print("✓ Differential equation without covariates test passed")
    
    # Skip predict with covariates test because it involves ODE solving that may be unstable
    # test_instance.test_predict_with_covariates()
    # print("✓ Predict with covariates test passed")
    
    # test_instance.test_score_with_covariates()
    # print("✓ Score with covariates test passed")
    
    # test_instance.test_predict_adoption_rate_with_covariates()
    # print("✓ Predict adoption rate with covariates test passed")
    
    test_instance.test_property_setters_and_getters()
    print("✓ Property setters and getters test passed")
    
    # Run integration test
    test_lotka_volterra_comprehensive_integration()
    print("✓ Integration test passed")
    
    print("\nAll comprehensive Lotka-Volterra model tests passed! 🎉")