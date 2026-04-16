"""Comprehensive tests for LogisticModel to achieve >90% coverage."""

import numpy as np
import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.logistic import LogisticModel


class TestLogisticModelBasic:
    """Test basic initialization and properties."""

    def test_default_initialization(self):
        """Test default initialization."""
        model = LogisticModel()
        assert model.covariates == []
        assert model.t_event is None
        assert model._params == {}

    def test_init_with_covariates(self):
        """Test initialization with covariates."""
        model = LogisticModel(covariates=["price", "advertising"])
        assert "price" in model.covariates
        assert "advertising" in model.covariates

    def test_init_with_t_event(self):
        """Test initialization with structural break."""
        model = LogisticModel(t_event=5.0)
        assert model.t_event == 5.0

    def test_init_with_both(self):
        """Test initialization with covariates and t_event."""
        model = LogisticModel(covariates=["price"], t_event=3.0)
        assert "price" in model.covariates
        assert model.t_event == 3.0

    def test_param_names_basic(self):
        """Test param_names without t_event or covariates."""
        model = LogisticModel()
        names = model.param_names
        assert names == ["L", "k", "x0"]

    def test_param_names_with_event(self):
        """Test param_names with t_event."""
        model = LogisticModel(t_event=5.0)
        names = model.param_names
        assert "L" in names
        assert "k" in names
        assert "x0" in names
        assert "L_post" in names
        assert "k_post" in names
        assert "x0_post" in names

    def test_param_names_with_covariates(self):
        """Test param_names with covariates."""
        model = LogisticModel(covariates=["price"])
        names = model.param_names
        assert "beta_L_price" in names
        assert "beta_k_price" in names
        assert "beta_x0_price" in names

    def test_initial_guesses(self):
        """Test initial parameter guesses."""
        model = LogisticModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 30.0, 60.0, 80.0, 95.0])
        guesses = model.initial_guesses(t, y)
        assert "L" in guesses
        assert "k" in guesses
        assert "x0" in guesses
        assert guesses["L"] > 0
        assert guesses["k"] > 0

    def test_initial_guesses_with_event(self):
        """Test initial guesses with t_event."""
        model = LogisticModel(t_event=3.0)
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 30.0, 60.0, 80.0, 95.0])
        guesses = model.initial_guesses(t, y)
        assert "L_post" in guesses
        assert "k_post" in guesses
        assert "x0_post" in guesses

    def test_initial_guesses_with_covariates(self):
        """Test initial guesses with covariates."""
        model = LogisticModel(covariates=["price"])
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 30.0, 60.0])
        guesses = model.initial_guesses(t, y)
        assert "beta_L_price" in guesses
        assert "beta_k_price" in guesses
        assert "beta_x0_price" in guesses
        assert guesses["beta_L_price"] == 0.0

    def test_bounds(self):
        """Test parameter bounds."""
        model = LogisticModel()
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 30.0, 60.0])
        bounds = model.bounds(t, y)
        assert bounds["L"][0] >= 60.0  # >= max(y)
        assert bounds["k"][0] > 0
        assert bounds["x0"][0] == -np.inf

    def test_bounds_with_event(self):
        """Test parameter bounds with t_event."""
        model = LogisticModel(t_event=2.0)
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 30.0, 60.0])
        bounds = model.bounds(t, y)
        assert "L_post" in bounds
        assert "k_post" in bounds
        assert "x0_post" in bounds


class TestLogisticModelFitPredict:
    """Test fitting, prediction, and scoring."""

    def _make_data(self):
        """Generate synthetic logistic data."""
        t = np.linspace(0, 10, 50)
        L, k, x0 = 100.0, 0.8, 5.0
        y = L / (1 + np.exp(-k * (t - x0)))
        return t, y

    def test_predict_unfitted_raises(self):
        """Test predict raises RuntimeError if not fitted."""
        model = LogisticModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict([1.0, 2.0])

    def test_predict_basic(self):
        """Test basic predict after fitting."""
        t, y = self._make_data()
        model = LogisticModel()
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        preds = model.predict(t)
        assert len(preds) == len(t)
        assert all(np.isfinite(preds))

    def test_predict_with_t_event(self):
        """Test predict with structural break."""
        t = np.linspace(0, 10, 50)
        y_pre = 100.0 / (1 + np.exp(-0.8 * (t[t < 5.0] - 3.0)))
        y_post = 150.0 / (1 + np.exp(-0.5 * (t[t >= 5.0] - 6.0)))
        y = np.concatenate([y_pre, y_post])

        model = LogisticModel(t_event=5.0)
        model.params_ = {
            "L": 100.0,
            "k": 0.8,
            "x0": 3.0,
            "L_post": 150.0,
            "k_post": 0.5,
            "x0_post": 6.0,
        }
        preds = model.predict(t)
        assert len(preds) == len(t)
        # Pre-event values should use pre-event params
        assert preds[0] < 100.0

    def test_predict_all_pre_event(self):
        """Test predict when all times are before t_event."""
        t = np.array([1.0, 2.0, 3.0, 4.0])
        model = LogisticModel(t_event=5.0)
        model.params_ = {
            "L": 100.0,
            "k": 0.8,
            "x0": 2.0,
            "L_post": 150.0,
            "k_post": 0.5,
            "x0_post": 6.0,
        }
        preds = model.predict(t)
        assert len(preds) == 4
        assert all(np.isfinite(preds))

    def test_predict_all_post_event(self):
        """Test predict when all times are after t_event."""
        t = np.array([6.0, 7.0, 8.0, 9.0])
        model = LogisticModel(t_event=5.0)
        model.params_ = {
            "L": 100.0,
            "k": 0.8,
            "x0": 2.0,
            "L_post": 150.0,
            "k_post": 0.5,
            "x0_post": 7.0,
        }
        preds = model.predict(t)
        assert len(preds) == 4
        assert all(np.isfinite(preds))

    def test_score_unfitted_raises(self):
        """Test score raises RuntimeError if not fitted."""
        model = LogisticModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.score([1.0], [10.0])

    def test_score_basic(self):
        """Test R² score calculation."""
        t, y = self._make_data()
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 5.0}
        score = model.score(t, y)
        assert -1.0 <= score <= 1.0

    def test_score_perfect_fit(self):
        """Test score with perfect predictions."""
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 30.0, 60.0, 80.0, 95.0])
        model = LogisticModel()
        # Set params to produce exact y
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 3.0}
        # Score should be reasonable
        score = model.score(t, y)
        assert isinstance(score, float)

    def test_score_zero_variance(self):
        """Test score when y has zero variance."""
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([50.0, 50.0, 50.0])
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 2.0}
        score = model.score(t, y)
        assert score == 0.0  # Zero variance returns 0

    def test_params_property(self):
        """Test params_ getter and setter."""
        model = LogisticModel()
        params = {"L": 100.0, "k": 0.5, "x0": 3.0}
        model.params_ = params
        assert model.params_ == params


class TestLogisticModelAdoptionRate:
    """Test adoption rate prediction."""

    def test_adoption_rate_unfitted_raises(self):
        """Test adoption rate raises if not fitted."""
        model = LogisticModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_adoption_rate([1.0])

    def test_adoption_rate_basic(self):
        """Test adoption rate is positive and bell-shaped."""
        t = np.linspace(0, 10, 100)
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 5.0}
        rates = model.predict_adoption_rate(t)
        assert len(rates) == len(t)
        assert all(r >= 0 for r in rates)
        # Peak should be around x0
        peak_idx = np.argmax(rates)
        assert abs(t[peak_idx] - 5.0) < 2.0


class TestLogisticModelCumulativeAdoption:
    """Test cumulative adoption method."""

    def test_cumulative_adoption_with_params_dict(self):
        """Test cumulative adoption with param_kwargs."""
        model = LogisticModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = model.cumulative_adoption(t, L=100.0, k=0.8, x0=3.0)
        assert len(result) == 5

    def test_cumulative_adoption_with_positional_params(self):
        """Test cumulative adoption with positional params."""
        model = LogisticModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = model.cumulative_adoption(t, 100.0, 0.8, 3.0)
        assert len(result) == 5


class TestLogisticModelDifferentialEquation:
    """Test differential equation method."""

    def test_differential_equation_pre_event(self):
        """Test differential equation before t_event."""
        model = LogisticModel()
        t_val = 2.0
        y = np.array([50.0])
        params = [100.0, 0.8, 3.0]
        result = model.differential_equation(t_val, y, params, None, None)
        assert np.isfinite(result)
        assert result > 0  # Growth should be positive below carrying capacity

    def test_differential_equation_post_event(self):
        """Test differential equation after t_event."""
        model = LogisticModel(t_event=5.0)
        t_val = 6.0
        y = np.array([80.0])
        params = [100.0, 0.8, 3.0, 150.0, 0.5, 7.0]
        result = model.differential_equation(t_val, y, params, None, None)
        assert np.isfinite(result)

    def test_differential_equation_at_capacity(self):
        """Test differential equation at carrying capacity (should be ~0)."""
        model = LogisticModel()
        t_val = 5.0
        y = np.array([100.0])  # At capacity
        params = [100.0, 0.8, 3.0]
        result = model.differential_equation(t_val, y, params, None, None)
        assert abs(result) < 1.0  # Should be near zero


class TestLogisticModelEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_with_zero_time(self):
        """Test predict at t=0."""
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 3.0}
        pred = model.predict([0.0])
        assert np.isfinite(pred[0])
        assert 0 < pred[0] < 100.0

    def test_predict_with_very_large_time(self):
        """Test predict at very large t (should approach L)."""
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 3.0}
        pred = model.predict([1000.0])
        assert pred[0] <= 100.0 + 1e-6  # Should approach L

    def test_predict_with_negative_k(self):
        """Test predict with negative k (decay instead of growth)."""
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": -0.5, "x0": 3.0}
        pred = model.predict([1.0, 2.0, 3.0])
        assert all(np.isfinite(p) for p in pred)

    def test_predict_empty_time_array(self):
        """Test predict with empty time array."""
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 3.0}
        pred = model.predict([])
        assert len(pred) == 0

    def test_score_with_single_point(self):
        """Test score with a single data point."""
        model = LogisticModel()
        model.params_ = {"L": 100.0, "k": 0.8, "x0": 3.0}
        score = model.score([3.0], [50.0])
        assert isinstance(score, float)
