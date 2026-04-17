# tests/test_plots_diagnostics.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.diagnostics_contract import DiagnosticsContract, UncertaintySummary
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.plots.diagnostics import plot_residuals


@pytest.fixture
def fitted_logistic_model():
    """Provides a fitted LogisticModel for testing."""
    t = np.linspace(0, 20, 100)
    y = 1.0 / (1 + np.exp(-1.5 * (t - 10.0))) + np.random.normal(0, 0.01, len(t))

    model = LogisticModel()
    fitter = ScipyFitter()
    fitter.fit(model, t, y)

    return model, t, y


def test_plot_residuals_runs_without_error(fitted_logistic_model):
    """Tests that plot_residuals runs without raising an exception."""
    model, t, y = fitted_logistic_model
    try:
        fig, _ = plot_residuals(model, t, y, show=False)
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"plot_residuals raised an exception: {e}")


def test_plot_residuals_raises_error_for_unfitted_model():
    """Tests that plot_residuals raises a RuntimeError for an unfitted model."""
    model = LogisticModel()
    t = np.linspace(0, 1, 10)
    y = np.random.rand(10)
    with pytest.raises(RuntimeError, match="Model has not been fitted yet."):
        plot_residuals(model, t, y)


def test_plot_residuals_accepts_diagnostics_contract(fitted_logistic_model):
    """Tests that plot_residuals can reuse a diagnostics contract."""
    model, t, y = fitted_logistic_model
    contract = DiagnosticsContract(
        residuals=np.asarray(y) - np.asarray(model.predict(t)),
        uncertainty=UncertaintySummary.point_estimate(),
        support_level="supported",
        provenance="deterministic",
        model_name="Logistic",
    )

    fig, axes = plot_residuals(model, t, y, diagnostics=contract, show=False)
    try:
        assert fig._suptitle.get_text() == "Residual Analysis"
        assert "support=supported" in axes[0].texts[0].get_text()
    finally:
        plt.close(fig)
