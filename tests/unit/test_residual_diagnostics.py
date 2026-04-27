import numpy as np
import pytest

from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.residual_analysis import ResidualAnalysis, analyze_residuals
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.utils.model_evaluation import (
    compute_residuals,
    residual_acf,
    residual_pacf,
)


def test_residual_functions():
    t = np.linspace(0, 10, 50)
    true_model = LogisticModel()
    true_model.params_ = {"L": 100.0, "k": 1.0, "x0": 5.0}
    y = true_model.predict(t)

    # add small noise
    y_noisy = y + np.random.normal(0, 1.0, size=len(t))

    # fit with ScipyFitter
    model = LogisticModel()
    fitter = ScipyFitter()
    fitter.fit(model, t, y_noisy)

    residuals = compute_residuals(model, t, y_noisy)
    acf_vals = residual_acf(model, t, y_noisy, nlags=5)
    pacf_vals = residual_pacf(model, t, y_noisy, nlags=5)

    assert residuals.shape[0] == len(t)
    assert len(acf_vals) == 6  # includes lag 0
    assert len(pacf_vals) == 6


def test_residual_analysis_summary_and_serialization():
    analysis = ResidualAnalysis(
        residuals=np.array([1.0, -1.0]),
        standardized_residuals=np.array([1.0, -1.0]),
        durbin_watson=2.0,
        shapiro_wilk_p=0.6,
        breusch_pagan_p=None,
        mean_residual=0.0,
        std_residual=1.0,
        max_abs_residual=1.0,
        residual_autocorrelation=np.array([0.0, 1.0, 0.0]),
    )

    assert analysis.has_autocorrelation() is False
    assert analysis.is_normally_distributed() is True
    assert analysis.has_heteroscedasticity() is False

    payload = analysis.to_dict()

    assert payload["residuals"] == [1.0, -1.0]
    assert payload["standardized_residuals"] == [1.0, -1.0]
    assert "Residual Analysis Summary" in analysis.summary()


def test_analyze_residuals_branch_coverage(monkeypatch: pytest.MonkeyPatch):
    single = analyze_residuals(np.array([1.0]))
    assert single.durbin_watson == 2.0
    assert single.residual_autocorrelation.tolist() == [0.0, 1.0, 0.0]
    assert np.isnan(single.shapiro_wilk_p)

    original_linregress = None

    def fail_linregress(*args, **kwargs):
        raise RuntimeError("forced failure")

    from innovate.fitters import residual_analysis as residual_analysis_module

    original_linregress = residual_analysis_module.stats.linregress
    monkeypatch.setattr(residual_analysis_module.stats, "linregress", fail_linregress)
    failed = analyze_residuals(
        np.array([1.0, -1.0, 2.0, -2.0, 3.0]),
        fitted_values=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    assert failed.breusch_pagan_p is None

    monkeypatch.setattr(residual_analysis_module.stats, "linregress", original_linregress)

    large = analyze_residuals(np.linspace(-1.0, 1.0, 5001))
    assert large.shapiro_wilk_p >= 0.0
