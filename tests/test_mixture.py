import numpy as np
import pytest
from innovate.models.mixture import MixtureModel
from innovate.diffuse.logistic import LogisticModel


def test_mixture_model_predict():
    # Create two logistic models for two segments
    model1 = LogisticModel()
    model2 = LogisticModel()

    # Create a mixture model
    mixture = MixtureModel(models=[model1, model2], weights=[0.4, 0.6])

    # Set parameters for the individual models
    model1.params_ = {"L": 1000, "k": 0.1, "x0": 20}
    model2.params_ = {"L": 2000, "k": 0.05, "x0": 30}
    params = {
        "L_1": 1000,
        "k_1": 0.1,
        "x0_1": 20,
        "L_2": 2000,
        "k_2": 0.05,
        "x0_2": 30,
        "weight_1": 0.4,
    }
    mixture.params_ = params

    t = np.linspace(0, 50, 100)

    # Check that the predictions are the weighted sum of the individual model predictions
    expected_predictions = 0.4 * np.array(model1.predict(t)) + 0.6 * np.array(
        model2.predict(t)
    )

    predictions = mixture.predict(t)

    assert np.allclose(predictions, expected_predictions)


def test_mixture_model_init():
    model1 = LogisticModel()
    # Test that weights must sum to 1
    with pytest.raises(ValueError):
        MixtureModel(models=[model1], weights=[0.5, 0.6])

    # Test that number of weights must match number of models
    with pytest.raises(ValueError):
        MixtureModel(models=[model1], weights=[0.5, 0.5])

    # Test that at least one model is required
    with pytest.raises(ValueError):
        MixtureModel(models=[], weights=[])
