"""
Tests for the robust BlackJAX-based BayesianFitter.

This test suite ensures the BayesianFitter works correctly and 
avoids the segmentation fault issues of the previous PyMC implementation.
"""

import numpy as np
import pytest

from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel  
from innovate.fitters.bayesian_fitter import BayesianFitter


class TestBayesianFitterRobust:
    """Test the robust BayesianFitter implementation."""
    
    def test_basic_fitting(self):
        """Test basic fitting functionality."""
        # Generate synthetic Bass model data
        t = np.linspace(1, 10, 20)
        p, q, m = 0.02, 0.3, 1000
        
        # Generate clean synthetic data
        true_adoption = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        y = true_adoption + np.random.normal(0, 0.01 * m, len(t))
        
        # Fit model
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=2, 
            num_warmup=100, 
            num_samples=100
        )
        
        # This should not cause segmentation fault
        fitter.fit(model, t, y)
        
        # Check that parameters were set
        assert model.params_ is not None
        assert 'p' in model.params_
        assert 'q' in model.params_
        assert 'm' in model.params_
        
        # Check parameter estimates are reasonable
        assert 0 < model.params_['p'] < 0.1
        assert 0 < model.params_['q'] < 1.0
        assert model.params_['m'] > 0
    
    def test_parameter_estimation_methods(self):
        """Test parameter estimation and uncertainty quantification methods."""
        t = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 25, 45, 70, 90])
        
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=2,
            num_warmup=50,
            num_samples=50
        )
        fitter.fit(model, t, y)
        
        # Test parameter estimates
        estimates = fitter.get_parameter_estimates()
        assert isinstance(estimates, dict)
        assert all(param in estimates for param in ['p', 'q', 'm'])
        assert all(isinstance(val, float) for val in estimates.values())
        
        # Test confidence intervals
        intervals = fitter.get_confidence_intervals()
        assert isinstance(intervals, dict)
        assert all(param in intervals for param in ['p', 'q', 'm'])
        for param, (lower, upper) in intervals.items():
            assert lower < upper
            assert lower <= estimates[param] <= upper
        
        # Test summary
        summary = fitter.get_summary()
        assert isinstance(summary, dict)
        for param in ['p', 'q', 'm']:
            assert param in summary
            param_summary = summary[param]
            assert 'mean' in param_summary
            assert 'std' in param_summary
            assert '2.5%' in param_summary
            assert '97.5%' in param_summary
    
    def test_different_models(self):
        """Test fitting with different model types."""
        t = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Test with LogisticModel
        # Generate logistic data: L / (1 + exp(-k*(t-x0)))
        L, k, x0 = 100, 0.5, 4
        y_logistic = L / (1 + np.exp(-k * (t - x0)))
        y_logistic += np.random.normal(0, 1, len(t))
        
        logistic_model = LogisticModel()
        fitter = BayesianFitter(
            num_chains=2,
            num_warmup=50, 
            num_samples=50
        )
        
        # Should fit without errors
        fitter.fit(logistic_model, t, y_logistic)
        
        # Check reasonable parameter estimates
        estimates = fitter.get_parameter_estimates()
        assert estimates['L'] > 0
        assert estimates['k'] > 0
        assert estimates['x0'] > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        t = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        
        fitter = BayesianFitter(
            num_chains=1,
            num_warmup=10,
            num_samples=10
        )
        
        # Test with unfitted model
        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            fitter.get_parameter_estimates()
        
        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            fitter.get_confidence_intervals()
        
        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            fitter.get_summary()
    
    def test_noisy_data_robustness(self):
        """Test robustness with very noisy data."""
        t = np.linspace(1, 10, 15)
        
        # Generate data with high noise
        p, q, m = 0.01, 0.2, 500
        true_adoption = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        # Add 20% noise
        y = true_adoption + np.random.normal(0, 0.2 * np.mean(true_adoption), len(t))
        
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=2,
            num_warmup=50,
            num_samples=50
        )
        
        # Should handle noisy data without crashing
        fitter.fit(model, t, y)
        
        # Should still produce reasonable estimates
        estimates = fitter.get_parameter_estimates()
        assert all(np.isfinite(val) for val in estimates.values())
        assert estimates['m'] > 0
    
    def test_small_dataset(self):
        """Test with minimal dataset."""
        t = np.array([1, 2, 3])
        y = np.array([5, 15, 30])
        
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=1,
            num_warmup=20,
            num_samples=20
        )
        
        # Should handle small datasets
        fitter.fit(model, t, y)
        
        estimates = fitter.get_parameter_estimates()
        assert all(np.isfinite(val) for val in estimates.values())
    
    def test_zero_adoption_data(self):
        """Test with edge case of zero adoption data."""
        t = np.array([1, 2, 3, 4, 5])
        y = np.array([0, 0, 1, 2, 3])  # Very slow initial adoption
        
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=1,
            num_warmup=20,
            num_samples=20
        )
        
        # Should handle zero values gracefully
        fitter.fit(model, t, y)
        
        estimates = fitter.get_parameter_estimates()
        # Parameters should be finite and positive
        assert all(np.isfinite(val) and val > 0 for val in estimates.values())
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        t = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 35, 55, 80])
        
        model1 = BassModel()
        fitter1 = BayesianFitter(
            num_chains=1,
            num_warmup=30,
            num_samples=30,
            random_seed=42
        )
        fitter1.fit(model1, t, y)
        estimates1 = fitter1.get_parameter_estimates()
        
        model2 = BassModel()
        fitter2 = BayesianFitter(
            num_chains=1,
            num_warmup=30,
            num_samples=30,
            random_seed=42
        )
        fitter2.fit(model2, t, y)
        estimates2 = fitter2.get_parameter_estimates()
        
        # Results should be very similar (allowing for small numerical differences)
        for param in estimates1:
            assert abs(estimates1[param] - estimates2[param]) < 0.1
    
    def test_performance_with_larger_dataset(self):
        """Test performance with a reasonably sized dataset."""
        t = np.linspace(1, 20, 50)
        
        # Generate realistic diffusion curve
        p, q, m = 0.015, 0.25, 2000
        true_adoption = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        y = true_adoption + np.random.normal(0, 0.05 * np.mean(true_adoption), len(t))
        
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=2,
            num_warmup=100,
            num_samples=100
        )
        
        # Should complete in reasonable time without errors
        fitter.fit(model, t, y)
        
        # Check quality of fit
        estimates = fitter.get_parameter_estimates()
        intervals = fitter.get_confidence_intervals()
        
        # Parameters should be in reasonable range
        assert 0.001 < estimates['p'] < 0.1
        assert 0.1 < estimates['q'] < 1.0
        assert 1000 < estimates['m'] < 5000
        
        # Confidence intervals should be meaningful
        for param in ['p', 'q', 'm']:
            lower, upper = intervals[param]
            width = upper - lower
            assert width > 0
            assert width < 10 * estimates[param]  # Not too wide
    
    @pytest.mark.slow
    def test_visualization_methods(self):
        """Test visualization methods (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Matplotlib not available for visualization tests")
        
        t = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([5, 15, 30, 50, 75, 95])
        
        model = BassModel()
        fitter = BayesianFitter(
            num_chains=2,
            num_warmup=50,
            num_samples=50
        )
        fitter.fit(model, t, y)
        
        # Test that plotting methods don't crash
        try:
            fitter.plot_trace()
            plt.close('all')
            
            fitter.plot_posterior()
            plt.close('all')
        except Exception as e:
            pytest.fail(f"Visualization methods failed: {e}")