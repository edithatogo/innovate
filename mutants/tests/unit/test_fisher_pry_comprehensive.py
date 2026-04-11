"""Tests for substitute/fisher_pry model - comprehensive coverage."""

import numpy as np
import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.substitute.fisher_pry import FisherPryModel


class TestFisherPryModelComprehensive:
    """Comprehensive tests for FisherPryModel."""

    def test_init_default(self):
        """Test default initialization."""
        model = FisherPryModel()
        assert model._params == {}

    def test_param_names(self):
        """Test param_names returns alpha and t0."""
        model = FisherPryModel()
        assert model.param_names == ["alpha", "t0"]

    def test_initial_guesses(self):
        """Test initial guesses generation."""
        model = FisherPryModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        guesses = model.initial_guesses(t, y)
        assert "alpha" in guesses
        assert "t0" in guesses
        assert guesses["alpha"] >= 0

    def test_initial_guesses_midpoint_t0(self):
        """Test that t0 guess is near the midpoint of transition."""
        model = FisherPryModel()
        t = np.linspace(0, 10, 20)
        # Sigmoid centered at t=5
        y = 1 / (1 + np.exp(-0.5 * (t - 5)))
        guesses = model.initial_guesses(t, y)
        assert 3.0 <= guesses["t0"] <= 7.0  # Should be near 5

    def test_bounds(self):
        """Test parameter bounds."""
        model = FisherPryModel()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        bounds = model.bounds(t, y)
        assert "alpha" in bounds
        assert "t0" in bounds
        assert bounds["alpha"][0] >= 0  # alpha >= 0

    def test_predict_unfitted_raises(self):
        """Test predict raises if not fitted."""
        model = FisherPryModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict([1.0, 2.0])

    def test_predict_basic(self):
        """Test basic prediction."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        t = np.linspace(0, 10, 20)
        preds = model.predict(t)
        assert len(preds) == 20
        assert all(0 <= p <= 1 for p in preds)  # Should be between 0 and 1

    def test_predict_monotonic(self):
        """Test predictions are monotonically non-decreasing."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        t = np.linspace(0, 10, 50)
        preds = model.predict(t)
        for i in range(1, len(preds)):
            assert preds[i] >= preds[i - 1] - 1e-10

    def test_predict_at_t0(self):
        """Test prediction at t0 should be 0.5."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        pred = model.predict([5.0])
        assert abs(pred[0] - 0.5) < 0.01

    def test_predict_early_time(self):
        """Test prediction at early time should be near 0."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        pred = model.predict([0.0])
        assert pred[0] < 0.1

    def test_predict_late_time(self):
        """Test prediction at late time should be near 1."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        pred = model.predict([20.0])
        assert pred[0] > 0.9

    def test_score_unfitted_raises(self):
        """Test score raises if not fitted."""
        model = FisherPryModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.score([1.0], [0.5])

    def test_score_basic(self):
        """Test score calculation."""
        model = FisherPryModel()
        t = np.linspace(0, 10, 20)
        y = 1 / (1 + np.exp(-0.5 * (t - 5.0)))
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        score = model.score(t, y)
        assert score > 0.99

    def test_params_property(self):
        """Test params_ getter/setter."""
        model = FisherPryModel()
        params = {"alpha": 0.5, "t0": 5.0}
        model.params_ = params
        assert model.params_ == params


class TestFisherPryModelEdgeCases:
    """Edge case tests for FisherPryModel."""

    def test_predict_single_point(self):
        """Test predict at single time point."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.5, "t0": 5.0}
        pred = model.predict([5.0])
        assert abs(pred[0] - 0.5) < 0.01

    def test_predict_with_zero_alpha(self):
        """Test predict with alpha=0 (no substitution)."""
        model = FisherPryModel()
        model.params_ = {"alpha": 0.0, "t0": 5.0}
        t = np.linspace(0, 10, 10)
        preds = model.predict(t)
        # With alpha=0, should be constant
        assert all(abs(p - preds[0]) < 0.01 for p in preds)

    def test_predict_with_very_large_alpha(self):
        """Test predict with very large alpha (step-like transition)."""
        model = FisherPryModel()
        model.params_ = {"alpha": 10.0, "t0": 5.0}
        t = np.array([4.0, 5.0, 6.0])
        preds = model.predict(t)
        assert preds[0] < 0.5  # Before t0
        assert preds[1] >= 0.49  # At t0
        assert preds[2] > 0.5  # After t0

    def test_predict_with_negative_alpha(self):
        """Test predict with negative alpha (reverse substitution)."""
        model = FisherPryModel()
        model.params_ = {"alpha": -0.5, "t0": 5.0}
        t = np.array([0.0, 5.0, 10.0])
        preds = model.predict(t)
        assert all(np.isfinite(p) for p in preds)
