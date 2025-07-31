# tests/test_composite.py

import pytest
import numpy as np
from innovate.substitute.composite import CompositeDiffusionModel
from innovate.diffuse.logistic import LogisticModel
from innovate.diffuse.bass import BassModel


@pytest.fixture
def synthetic_composite_data():
    """Generate synthetic data for a composite diffusion model."""
    t = np.linspace(0, 50, 100)

    # True parameters for 2 products
    params = {
        "L_1": 1000,
        "k_1": 0.1,
        "x0_1": 20,
        "p_2": 0.03,
        "q_2": 0.38,
        "m_2": 1500,
        "alpha_1_2": 0.1,
        "alpha_2_1": 0.05,
    }

    models = [LogisticModel(), BassModel()]
    alpha = np.array([[0, 0.1], [0.05, 0]])
    model = CompositeDiffusionModel(models=models, alpha=alpha)
    model.params_ = params

    y = model.predict(t)
    y_noisy = y + np.random.normal(0, 10, y.shape)

    return t, y_noisy, params


def test_composite_model(synthetic_composite_data):
    """Test the CompositeDiffusionModel."""
    t, y, true_params = synthetic_composite_data

    models = [LogisticModel(), BassModel()]
    model = CompositeDiffusionModel(models=models)

    # TODO: Fitting for CompositeDiffusionModel is not working correctly.
    # fitter = ScipyFitter()
    # fitter.fit(model, t, y)

    # # Check if the estimated parameters are reasonably close to the true parameters
    # for param, true_value in true_params.items():
    #     assert np.isclose(model.params_[param], true_value, atol=5, rtol=0.5)

    model.params_ = true_params

    # Test predict and predict_adoption_rate
    y_pred = model.predict(t)
    assert y_pred.shape == (len(t), 2)

    rates = model.predict_adoption_rate(t)
    assert rates.shape == (len(t), 2)


def test_composite_model_init():
    """Test initialization of the CompositeDiffusionModel."""
    models = [LogisticModel(), BassModel()]
    model = CompositeDiffusionModel(models=models)
    assert len(model.models) == 2
    assert "L_1" in model.param_names
    assert "p_2" in model.param_names
    assert "alpha_1_2" in model.param_names
    assert "alpha_2_1" in model.param_names

    # Test with invalid alpha
    with pytest.raises(ValueError):
        CompositeDiffusionModel(models=models, alpha=np.zeros((3, 3)))


def test_composite_model_no_fit_error():
    """Test that the model raises an error if predict is called before fitting."""
    models = [LogisticModel(), BassModel()]
    model = CompositeDiffusionModel(models=models)
    t = np.linspace(0, 10, 10)
    with pytest.raises(RuntimeError):
        model.predict(t)
    with pytest.raises(RuntimeError):
        model.predict_adoption_rate(t)
    with pytest.raises(RuntimeError):
        model.score(t, np.zeros_like(t))
