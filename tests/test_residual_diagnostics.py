import numpy as np
from innovate.diffuse.logistic import LogisticModel
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
