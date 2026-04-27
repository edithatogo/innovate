"""Tests for enhanced ScipyFitter with multiple optimization methods and diagnostics."""

import numpy as np
import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.scipy_fitter import FitDiagnostics, ScipyFitter


class TestFitDiagnostics:
    """Test FitDiagnostics dataclass."""

    def test_default_initialization(self):
        """Test default initialization of FitDiagnostics."""
        diag = FitDiagnostics()
        assert diag.r_squared == 0.0
        assert diag.rmse == 0.0
        assert diag.aic == 0.0
        assert diag.residuals.size == 0
        assert diag.fitted_params == {}

    def test_summary_format(self):
        """Test summary method produces formatted output."""
        diag = FitDiagnostics(
            r_squared=0.95,
            rmse=2.5,
            mae=1.8,
            aic=120.5,
            bic=125.3,
            n_observations=40,
            n_parameters=3,
            optimization_method="curve_fit",
            convergence_status="converged",
        )
        summary = diag.summary()
        assert "R²:" in summary
        assert "0.95" in summary
        assert "curve_fit" in summary
        assert "converged" in summary

    def test_to_dict_serializes_contract_fields(self):
        """Test structured serialization of fit diagnostics."""
        diag = FitDiagnostics(
            r_squared=0.95,
            rmse=2.5,
            mae=1.8,
            aic=120.5,
            bic=125.3,
            residuals=np.array([1.0, -1.0]),
            fitted_params={"p": 0.03},
            n_observations=40,
            n_parameters=3,
            optimization_method="curve_fit",
            convergence_status="converged",
        )

        payload = diag.to_dict()

        assert payload["residuals"] == [1.0, -1.0]
        assert payload["fitted_params"] == {"p": 0.03}
        assert payload["uncertainty"]["report_type"] == "point_estimate"
        assert payload["warnings"] == []


class TestScipyFitterMethods:
    """Test different optimization methods."""

    def _make_bass_data(self, p=0.03, q=0.38, m=100.0, noise=2.0, n=40):
        """Generate synthetic Bass model data."""
        t = np.linspace(0, 12, n)
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        np.random.seed(42)
        y += np.random.normal(0, noise, len(y))
        return t, np.maximum(y, 0)

    def test_curve_fit_method(self):
        """Test fitting with curve_fit method."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(method="curve_fit")
        fitter.fit(model, t, y)
        assert model.params_ is not None
        assert "p" in model.params_
        assert "q" in model.params_
        assert "m" in model.params_

    def test_least_squares_method(self):
        """Test fitting with least_squares method."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(method="least_squares")
        fitter.fit(model, t, y)
        assert model.params_ is not None
        assert all(p > 0 for p in model.params_.values())

    def test_nelder_mead_method(self):
        """Test fitting with Nelder-Mead method."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(method="nelder_mead")
        fitter.fit(model, t, y)
        assert model.params_ is not None

    def test_lbfgsb_method(self):
        """Test fitting with L-BFGS-B method."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(method="lbfgsb")
        fitter.fit(model, t, y)
        assert model.params_ is not None

    def test_differential_evolution_method(self):
        """Test fitting with differential evolution method."""
        t, y = self._make_bass_data(n=20)  # Smaller dataset for speed
        model = BassModel()
        fitter = ScipyFitter(method="differential_evolution", maxiter=50)
        fitter.fit(model, t, y)
        assert model.params_ is not None

    def test_auto_method_selection(self):
        """Test automatic method selection."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(method="auto")
        fitter.fit(model, t, y)
        assert model.params_ is not None
        # Auto should select curve_fit for this dataset size
        assert fitter.diagnostics is not None

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(method="unknown_method")  # type: ignore
        with pytest.raises(ValueError, match="Unknown method"):
            fitter.fit(model, t, y)


class TestScipyFitterDiagnostics:
    """Test fit diagnostics functionality."""

    def _make_bass_data(self, p=0.03, q=0.38, m=100.0, noise=2.0):
        """Generate synthetic Bass model data."""
        t = np.linspace(0, 12, 40)
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        np.random.seed(42)
        y += np.random.normal(0, noise, len(y))
        return t, np.maximum(y, 0)

    def test_diagnostics_stored_after_fit(self):
        """Test that diagnostics are stored after fitting."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(store_diagnostics=True)
        fitter.fit(model, t, y)
        assert fitter.diagnostics is not None

    def test_diagnostics_not_stored_when_disabled(self):
        """Test that diagnostics are not stored when disabled."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter(store_diagnostics=False)
        fitter.fit(model, t, y)
        assert fitter.diagnostics is None

    def test_r_squared_is_high_for_good_fit(self):
        """Test that R² is high for a good fit."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert fitter.diagnostics.r_squared > 0.8

    def test_rmse_is_positive_and_reasonable(self):
        """Test that RMSE is positive and reasonable."""
        t, y = self._make_bass_data(noise=2.0)
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert 0 < fitter.diagnostics.rmse < 10

    def test_aic_bic_are_finite(self):
        """Test that AIC and BIC are finite."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert np.isfinite(fitter.diagnostics.aic)
        assert np.isfinite(fitter.diagnostics.bic)

    def test_residuals_length_matches_data(self):
        """Test that residuals have same length as input data."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert len(fitter.diagnostics.residuals) == len(y)

    def test_diagnostics_summary(self):
        """Test diagnostics summary method."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        summary = fitter.diagnostics.summary()
        assert isinstance(summary, str)
        assert "R²" in summary
        assert "RMSE" in summary


class TestScipyFitterValidation:
    """Test input validation for ScipyFitter."""

    def test_empty_arrays_raises(self):
        """Test that empty arrays raise ValueError."""
        model = BassModel()
        fitter = ScipyFitter()
        with pytest.raises(ValueError, match="must not be empty"):
            fitter.fit(model, [], [])

    def test_mismatched_lengths_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        model = BassModel()
        fitter = ScipyFitter()
        with pytest.raises(ValueError, match="same length"):
            fitter.fit(model, [1, 2, 3], [1, 2])

    def test_nan_in_y_raises(self):
        """Test that NaN in y raises ValueError."""
        model = BassModel()
        fitter = ScipyFitter()
        with pytest.raises(ValueError, match="non-finite"):
            fitter.fit(model, [1, 2, 3], [1, float("nan"), 3])

    def test_inf_in_t_raises(self):
        """Test that Inf in t raises ValueError."""
        model = BassModel()
        fitter = ScipyFitter()
        with pytest.raises(ValueError, match="non-finite"):
            fitter.fit(model, [1, float("inf"), 3], [1, 2, 3])

    def test_weights_parameter_accepted(self):
        """Test that weights parameter is accepted."""
        t = np.linspace(0, 12, 40)
        p, q, m = 0.03, 0.38, 100.0
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        np.random.seed(42)
        y += np.random.normal(0, 2, len(y))
        y = np.maximum(y, 0)

        model = BassModel()
        fitter = ScipyFitter()
        weights = np.ones(len(t))
        fitter.fit(model, t, y, weights=weights)
        assert model.params_ is not None


class TestScipyFitterDifferentModels:
    """Test ScipyFitter with different model types."""

    def _make_data(self, model_type="bass"):
        """Generate synthetic data for different model types."""
        t = np.linspace(0, 12, 40)
        np.random.seed(42)

        if model_type == "bass":
            p, q, m = 0.03, 0.38, 100.0
            y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
            y += np.random.normal(0, 2, len(y))
        elif model_type == "gompertz":
            y = 100.0 * np.exp(-2.0 * np.exp(-0.3 * t))
            y += np.random.normal(0, 2, len(y))
        elif model_type == "logistic":
            y = 100.0 / (1 + np.exp(-0.8 * (t - 5.0)))
            y += np.random.normal(0, 2, len(y))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return t, np.maximum(y, 0)

    def test_fit_gompertz_model(self):
        """Test fitting Gompertz model."""
        t, y = self._make_data("gompertz")
        model = GompertzModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert model.params_ is not None
        assert fitter.diagnostics.r_squared > 0.5

    def test_fit_logistic_model(self):
        """Test fitting Logistic model."""
        t, y = self._make_data("logistic")
        model = LogisticModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        assert model.params_ is not None
        assert fitter.diagnostics.r_squared > 0.5

    def test_custom_bounds(self):
        """Test fitting with custom bounds."""
        t, y = self._make_data("bass")
        model = BassModel()
        fitter = ScipyFitter()
        custom_bounds = ([1e-5, 1e-5, 50], [0.5, 1.0, 200])
        fitter.fit(model, t, y, bounds=custom_bounds)
        assert model.params_["p"] >= 1e-5
        assert model.params_["m"] <= 200

    def test_custom_initial_guesses(self):
        """Test fitting with custom initial guesses."""
        t, y = self._make_data("bass")
        model = BassModel()
        fitter = ScipyFitter()
        custom_p0 = [0.05, 0.3, 150]
        fitter.fit(model, t, y, p0=custom_p0)
        assert model.params_ is not None
