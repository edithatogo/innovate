import numpy as np
from innovate.diffuse.bass import BassModel
from innovate.utils.model_evaluation import model_aic, model_bic
from innovate.utils.metrics import calculate_rss, calculate_aic, calculate_bic


def test_model_aic_bic():
    t = np.arange(0, 10)
    true_model = BassModel()
    true_model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
    y_true = true_model.predict(t)

    fitted_model = BassModel()
    fitted_model.params_ = {"p": 0.03, "q": 0.25, "m": 900}

    aic_value = model_aic(fitted_model, t, y_true)
    bic_value = model_bic(fitted_model, t, y_true)

    rss = calculate_rss(y_true, fitted_model.predict(t))
    n_samples = len(y_true)
    n_params = len(fitted_model.param_names) + 1
    expected_aic = calculate_aic(n_params, n_samples, rss)
    expected_bic = calculate_bic(n_params, n_samples, rss)

    assert np.isclose(aic_value, expected_aic)
    assert np.isclose(bic_value, expected_bic)
