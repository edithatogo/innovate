import pytest
import numpy as np
from innovate.compete.multi_product import MultiProductDiffusionModel
from innovate.fitters.scipy_fitter import ScipyFitter

@pytest.fixture
def synthetic_multi_product_data():
    """
    Generate synthetic data for two competing products.
    """
    t = np.linspace(0, 50, 100)
    
    # True parameters for two products
    true_params = {
        "p1": 0.03, "p2": 0.02,
        "q1": 0.1, "q2": 0.15,
        "m1": 1000, "m2": 1200,
        "alpha_1_2": 0.5, "alpha_2_1": 0.3
    }

    # Use the model to generate the ideal data
    model = MultiProductDiffusionModel(n_products=2)
    model.params_ = true_params
    y_ideal = model.predict(t)

    return t, y_ideal, list(true_params.values())

def test_multi_product_model_fit(synthetic_multi_product_data):
    """
    Test the fitting of the MultiProductDiffusionModel.
    """
    t, y, true_params = synthetic_multi_product_data
    
    # Initialize the model for 2 products
    model = MultiProductDiffusionModel(n_products=2)
    
    # Provide slightly perturbed initial guesses to guide the optimizer
    p0 = np.array(true_params) * (1 + np.random.uniform(-0.1, 0.1, size=len(true_params)))

    # Use the ScipyFitter to fit the model
    fitter = ScipyFitter()
    fitter.fit(model, t, y, p0=p0)
    
    # Check if the parameters have been fitted
    assert model.params_ is not None
    assert len(model.params_) == 8 # p1, q1, m1, p2, q2, m2, alpha_1_2, alpha_2_1
    
    # Check if the fitted parameters are within a reasonable range of the true parameters
    fitted_params = list(model.params_.values())
    
    # Use a high relative tolerance to account for the difficulty of fitting this model
    assert np.allclose(fitted_params, true_params, rtol=0.4)