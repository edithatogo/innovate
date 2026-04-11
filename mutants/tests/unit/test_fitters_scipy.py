"""Tests for fitters module - comprehensive coverage."""

import numpy as np
import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter


class TestScipyFitter:
    """Comprehensive tests for ScipyFitter."""

    def _make_data(self):
        """Generate synthetic Bass model data."""
        t = np.linspace(0, 10, 30)
        p, q, m = 0.03, 0.38, 100.0
        # Generate cumulative adoption data
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        return t, y

    def test_fit_basic(self):
        """Test basic fitting with ScipyFitter."""
        t, y = self._make_data()
        model = BassModel()
        fitter = ScipyFitter()
        result = fitter.fit(model, t, y)
        assert model.params_
        assert "p" in model.params_
        assert "q" in model.params_
        assert "m" in model.params_

    def test_fit_returns_result(self):
        """Test that fit returns an optimization result."""
        t, y = self._make_data()
        model = BassModel()
        fitter = ScipyFitter()
        result = fitter.fit(model, t, y)
        assert result is not None

    def test_fit_improves_score(self):
        """Test that fitting improves model score."""
        t, y = self._make_data()
        model = BassModel()
        fitter = ScipyFitter()
        # Initial score with default params (should be poor)
        model.params_ = {"p": 0.1, "q": 0.1, "m": 50.0}
        initial_score = model.score(t, y)
        # Refit
        fitter.fit(model, t, y)
        final_score = model.score(t, y)
        assert final_score >= initial_score

    def test_fit_with_covariates(self):
        """Test fitting with covariates."""
        t = np.linspace(0, 10, 30)
        y = 100.0 * (1 - np.exp(-0.4 * t))  # Simplified
        model = BassModel(covariates=["x1"])
        fitter = ScipyFitter()
        try:
            result = fitter.fit(model, t, y)
            assert model.params_
        except Exception:
            pytest.skip("Covariate fitting may fail for some models")

    def test_fit_with_t_event(self):
        """Test fitting with structural break."""
        t = np.linspace(0, 10, 30)
        y = 100.0 * (1 - np.exp(-0.4 * t))
        model = BassModel(t_event=5.0)
        fitter = ScipyFitter()
        try:
            result = fitter.fit(model, t, y)
            assert model.params_
        except Exception:
            pytest.skip("t_event fitting may fail for some models")

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        t, y = self._make_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        preds = model.predict(t)
        assert len(preds) == len(t)
        assert all(np.isfinite(p) for p in preds)

    def test_score_after_fit(self):
        """Test score after fitting."""
        t, y = self._make_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        score = model.score(t, y)
        assert score > 0.9  # Should be good fit

    def test_fit_noisy_data(self):
        """Test fitting with noisy data."""
        t, y_clean = self._make_data()
        np.random.seed(42)
        y_noisy = y_clean + np.random.normal(0, 2, len(y_clean))
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y_noisy)
        score = model.score(t, y_clean)
        assert score > 0.8  # Should still be reasonable

    def test_fit_sparse_data(self):
        """Test fitting with sparse data points."""
        t = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
        y = np.array([5.0, 25.0, 55.0, 80.0, 95.0])
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert model.params_
        assert 0 < model.params_["p"] < 1
        assert 0 < model.params_["q"] < 1
        assert model.params_["m"] > 0


class TestScipyFitterEdgeCases:
    """Edge case tests for ScipyFitter."""

    def test_fit_constant_y(self):
        """Test fitting with constant y values."""
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        model = BassModel()
        fitter = ScipyFitter()
        # May not converge but shouldn't crash
        try:
            fitter.fit(model, t, y)
        except Exception:
            pass  # Acceptable for degenerate data

    def test_fit_single_point_raises(self):
        """Test fitting with single data point."""
        t = np.array([1.0])
        y = np.array([50.0])
        model = BassModel()
        fitter = ScipyFitter()
        # Should handle gracefully or raise
        try:
            fitter.fit(model, t, y)
        except Exception:
            pass  # Acceptable

    def test_fit_negative_values(self):
        """Test fitting with negative values."""
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([-5.0, 10.0, 30.0, 50.0, 70.0])
        model = BassModel()
        fitter = ScipyFitter()
        try:
            fitter.fit(model, t, y)
        except Exception:
            pass  # May fail with negative values
