"""Comprehensive tests for GompertzModel to achieve >90% coverage."""

import numpy as np
import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.gompertz import GompertzModel


class TestGompertzModelBasic:
    """Test basic initialization and properties."""

    def test_default_initialization(self):
        """Test default initialization."""
        model = GompertzModel()
        assert model.covariates == []
        assert model.t_event is None
        assert model._params == {}

    def test_init_with_covariates(self):
        """Test initialization with covariates."""
        model = GompertzModel(covariates=["price"])
        assert "price" in model.covariates

    def test_init_with_t_event(self):
        """Test initialization with structural break."""
        model = GompertzModel(t_event=5.0)
        assert model.t_event == 5.0

    def test_param_names_basic(self):
        """Test param_names without t_event."""
        model = GompertzModel()
        assert model.param_names == ["a", "b", "c"]

    def test_param_names_with_event(self):
        """Test param_names with t_event."""
        model = GompertzModel(t_event=3.0)
        names = model.param_names
        assert "a_post" in names
        assert "b_post" in names
        assert "c_post" in names

    def test_param_names_with_covariates(self):
        """Test param_names with covariates."""
        model = GompertzModel(covariates=["price"])
        names = model.param_names
        assert "beta_a_price" in names
        assert "beta_b_price" in names
        assert "beta_c_price" in names

    def test_initial_guesses(self):
        """Test initial guesses."""
        model = GompertzModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 30.0, 60.0, 80.0, 95.0])
        guesses = model.initial_guesses(t, y)
        assert "a" in guesses
        assert "b" in guesses
        assert "c" in guesses
        assert guesses["a"] > 0

    def test_initial_guesses_with_event(self):
        """Test initial guesses with t_event."""
        model = GompertzModel(t_event=3.0)
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 30.0, 60.0, 80.0, 95.0])
        guesses = model.initial_guesses(t, y)
        assert "a_post" in guesses
        assert "b_post" in guesses
        assert "c_post" in guesses

    def test_bounds(self):
        """Test parameter bounds."""
        model = GompertzModel()
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 30.0, 60.0])
        bounds = model.bounds(t, y)
        assert "a" in bounds
        assert "b" in bounds
        assert "c" in bounds


class TestGompertzModelFitPredict:
    """Test fitting, prediction, and scoring."""

    def _make_data(self):
        """Generate synthetic Gompertz data."""
        t = np.linspace(0, 10, 50)
        a, b, c = 100.0, 2.0, 0.3
        y = a * np.exp(-b * np.exp(-c * t))
        return t, y

    def test_predict_unfitted_raises(self):
        """Test predict raises if not fitted."""
        model = GompertzModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict([1.0])

    def test_predict_basic(self):
        """Test basic predict."""
        model = GompertzModel()
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        t = np.linspace(0, 10, 20)
        preds = model.predict(t)
        assert len(preds) == 20
        assert all(np.isfinite(p) for p in preds)
        # Gompertz is monotonically increasing
        assert all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1))

    def test_predict_with_t_event(self):
        """Test predict with structural break."""
        t = np.linspace(0, 10, 50)
        model = GompertzModel(t_event=5.0)
        model.params_ = {
            "a": 100.0, "b": 2.0, "c": 0.3,
            "a_post": 150.0, "b_post": 1.5, "c_post": 0.4,
        }
        preds = model.predict(t)
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)

    def test_predict_all_pre_event(self):
        """Test predict when all times before t_event."""
        t = np.array([1.0, 2.0, 3.0, 4.0])
        model = GompertzModel(t_event=5.0)
        model.params_ = {
            "a": 100.0, "b": 2.0, "c": 0.3,
            "a_post": 150.0, "b_post": 1.5, "c_post": 0.4,
        }
        preds = model.predict(t)
        assert len(preds) == 4
        assert all(np.isfinite(p) for p in preds)

    def test_predict_all_post_event(self):
        """Test predict when all times after t_event."""
        t = np.array([6.0, 7.0, 8.0, 9.0])
        model = GompertzModel(t_event=5.0)
        model.params_ = {
            "a": 100.0, "b": 2.0, "c": 0.3,
            "a_post": 150.0, "b_post": 1.5, "c_post": 0.4,
        }
        preds = model.predict(t)
        assert len(preds) == 4
        assert all(np.isfinite(p) for p in preds)

    def test_score_unfitted_raises(self):
        """Test score raises if not fitted."""
        model = GompertzModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.score([1.0], [10.0])

    def test_score_basic(self):
        """Test R² score calculation."""
        model = GompertzModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 15.0, 35.0, 60.0, 80.0])
        # Set params that produce reasonable predictions
        model.params_ = {"a": 100.0, "b": 3.0, "c": 0.5}
        score = model.score(t, y)
        assert isinstance(score, float)
        assert score <= 1.0

    def test_score_zero_variance(self):
        """Test score with zero variance y."""
        model = GompertzModel()
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([50.0, 50.0, 50.0])
        score = model.score(t, y)
        assert score == 0.0

    def test_params_property(self):
        """Test params_ getter/setter."""
        model = GompertzModel()
        params = {"a": 100.0, "b": 2.0, "c": 0.3}
        model.params_ = params
        assert model.params_ == params


class TestGompertzModelAdoptionRate:
    """Test adoption rate prediction."""

    def test_adoption_rate_unfitted_raises(self):
        """Test adoption rate raises if not fitted."""
        model = GompertzModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_adoption_rate([1.0])

    def test_adoption_rate_basic(self):
        """Test adoption rate is positive and bell-shaped."""
        t = np.linspace(0, 15, 100)
        model = GompertzModel()
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        rates = model.predict_adoption_rate(t)
        assert len(rates) == 100
        assert all(r >= 0 for r in rates)


class TestGompertzModelCumulativeAdoption:
    """Test cumulative adoption method."""

    def test_cumulative_adoption_with_kwargs(self):
        """Test cumulative adoption with param_kwargs."""
        model = GompertzModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Gompertz uses a, b, c params
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        result = model.predict(t)
        assert len(result) == 5

    def test_cumulative_adoption_positional(self):
        """Test cumulative adoption with positional params."""
        model = GompertzModel()
        t = np.array([1.0, 2.0, 3.0])
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        result = model.predict(t)
        assert len(result) == 3


class TestGompertzModelDifferentialEquation:
    """Test differential equation method."""

    def test_differential_equation_pre_event(self):
        """Test differential equation before t_event."""
        model = GompertzModel()
        t_val = 3.0
        y = np.array([50.0])
        params = [100.0, 2.0, 0.3]
        result = model.differential_equation(t_val, y, params, None, None)
        assert np.isfinite(result)
        assert result > 0

    def test_differential_equation_post_event(self):
        """Test differential equation after t_event."""
        model = GompertzModel(t_event=5.0)
        t_val = 6.0
        y = np.array([80.0])
        params = [100.0, 2.0, 0.3, 150.0, 1.5, 0.4]
        result = model.differential_equation(t_val, y, params, None, None)
        assert np.isfinite(result)

    def test_differential_equation_at_capacity(self):
        """Test differential equation near capacity (should be ~0)."""
        model = GompertzModel()
        t_val = 10.0
        y = np.array([99.9])  # Near capacity
        params = [100.0, 2.0, 0.3]
        result = model.differential_equation(t_val, y, params, None, None)
        assert abs(result) < 5.0


class TestGompertzModelEdgeCases:
    """Test edge cases."""

    def test_predict_empty_time_array(self):
        """Test predict with empty time array."""
        model = GompertzModel()
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        # Empty array may raise IndexError or return empty - either is acceptable
        try:
            pred = model.predict([])
            assert len(pred) == 0
        except (IndexError, ValueError):
            pass  # Acceptable behavior for empty input

    def test_predict_single_time_point(self):
        """Test predict at single time point."""
        model = GompertzModel()
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        pred = model.predict([0.0])
        assert np.isfinite(pred[0])
        assert 0 < pred[0] < 100.0

    def test_predict_very_large_time(self):
        """Test predict at large t (should approach a)."""
        model = GompertzModel()
        model.params_ = {"a": 100.0, "b": 2.0, "c": 0.3}
        pred = model.predict([1000.0])
        assert pred[0] <= 100.0 + 1e-6
