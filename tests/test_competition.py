# tests/test_competition.py

import pytest
import numpy as np
import pandas as pd
from typing import Sequence
from innovate.compete.competition import MultiProductDiffusionModel
from innovate.substitute.fisher_pry import FisherPryModel
from innovate.diffuse.bass import BassModel

@pytest.fixture
def fitted_bass_model():
    """A fitted Bass model."""
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1.0}
    return model

@pytest.fixture
def fitted_fisher_pry_model():
    """A fitted Fisher-Pry model."""
    model = FisherPryModel()
    model.params_ = {"alpha": 0.5, "t0": 10}
    return model

def test_competition_model_init():
    """Test initialization of the MultiProductDiffusionModel."""
    p_vals = [0.02, 0.015]
    Q_matrix = [[0.3, 0.05], [0.03, 0.25]]
    m_vals = [1000, 800]
    product_names = ["ProdA", "ProdB"]
    model = MultiProductDiffusionModel(n_products=2, p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)
    assert model.n_products == 2
    assert len(model.p) == 2
    assert model.Q.shape == (2, 2)
    assert len(model.m) == 2
    assert model.names == product_names

def test_competition_model_init_empty():
    """Test that initialization with invalid parameters raises an error."""
    with pytest.raises(ValueError, match="Number of products must be at least 1."):
        MultiProductDiffusionModel(n_products=0)

def test_multi_product_model_fit_basic():
    """Test the fit method of the MultiProductDiffusionModel with basic parameters."""
    # Generate synthetic data for two products
    t = np.arange(1, 51)
    # Simulate Bass-like diffusion for two products with some interaction
    # These are not exact, just for generating plausible data
    y1_true = 1000 / (1 + np.exp(-0.3 * (t - 20))) # Logistic-like
    y2_true = 800 / (1 + np.exp(-0.25 * (t - 25))) # Logistic-like

    # Add some noise and ensure cumulative
    np.random.seed(42)
    y1_obs = np.maximum.accumulate(y1_true + np.random.normal(0, 20, len(t)))
    y2_obs = np.maximum.accumulate(y2_true + np.random.normal(0, 15, len(t)))
    y_obs = np.vstack([y1_obs, y2_obs]).T

    model = MultiProductDiffusionModel(n_products=2)
    model.fit(t, y_obs)

    assert model.params_ is not None
    assert len(model.params_) == len(model.param_names)

    predictions = model.predict(t, y_obs[0, :])
    assert predictions.shape == y_obs.shape
    assert np.all(predictions >= 0)

    score = model.score(t, y_obs, y_obs[0, :])
    assert score > 0.5 # Should be a reasonably good fit

    # Test predict_adoption_rate
    rates = model.predict_adoption_rate(t, y_obs[0, :])
    assert rates.shape == y_obs.shape
    assert np.all(rates >= -1e-6) # Rates can be very slightly negative due to noise/fitting, but should be near 0 or positive

def test_multi_product_model_fit_with_covariates():
    """Test the fit method of the MultiProductDiffusionModel with covariates."""
    t = np.arange(1, 51)
    # Simulate a covariate that increases over time
    cov_values = np.linspace(0.1, 1.0, len(t))

    # Simulate Bass-like diffusion for two products with some interaction
    # and a covariate influencing parameters
    # For simplicity, let's make 'p' and 'q' influenced by the covariate
    p1_true_base = 0.02
    q1_true_base = 0.3
    m1_true_base = 1000
    alpha12_true_base = 0.05

    p2_true_base = 0.015
    q2_true_base = 0.25
    m2_true_base = 800
    alpha21_true_base = 0.03

    # Define how covariate influences parameters (these are the 'true' beta coefficients)
    beta_p1_cov1_true = 0.005
    beta_q1_cov1_true = 0.1
    beta_m1_cov1_true = 100
    beta_alpha12_cov1_true = 0.02

    beta_p2_cov1_true = 0.003
    beta_q2_cov1_true = 0.08
    beta_m2_cov1_true = 80
    beta_alpha21_cov1_true = 0.01

    # Create a dummy model to generate data with time-varying parameters
    # This is a simplified way to get data that should be fit by the covariate model
    class DataGenModel(MultiProductDiffusionModel):
        def __init__(self, n_products: int, covariates: Sequence[str] = None):
            super().__init__(n_products, np.array([p1_true_base, p2_true_base]), np.array([[alpha12_true_base, 0], [0, alpha21_true_base]]), np.array([m1_true_base, m2_true_base]), ["ProdA", "ProdB"], covariates)
            self.true_betas = {
                "beta_p1_cov1": beta_p1_cov1_true, "beta_q1_cov1": beta_q1_cov1_true, "beta_m1_cov1": beta_m1_cov1_true,
                "beta_alpha_1_2_cov1": beta_alpha12_cov1_true,
                "beta_p2_cov1": beta_p2_cov1_true, "beta_q2_cov1": beta_q2_cov1_true, "beta_m2_cov1": beta_m2_cov1_true,
                "beta_alpha_2_1_cov1": beta_alpha21_cov1_true,
            }

        def differential_equation(self, t_val, y_val, params_flat, covariates_dict, t_eval):
            # Override to use true betas for data generation
            num_products = self.n_products
            
            p_base = np.array(params_flat[:num_products])
            q_base = np.array(params_flat[num_products:2*num_products])
            m_base = np.array(params_flat[2*num_products:3*num_products])
            alpha_base_flat = np.array(params_flat[3*num_products:])

            p_t = np.copy(p_base)
            q_t = np.copy(q_base)
            m_t = np.copy(m_base)
            alpha_t_flat = np.copy(alpha_base_flat)

            if covariates_dict:
                for cov_name in self.covariates:
                    cov_values_for_gen = covariates_dict[cov_name]
                    cov_val_t = np.interp(t_val, t_eval, cov_values_for_gen)
                    
                    p_t[0] += self.true_betas[f"beta_p1_{cov_name}"] * cov_val_t
                    q_t[0] += self.true_betas[f"beta_q1_{cov_name}"] * cov_val_t
                    m_t[0] += self.true_betas[f"beta_m1_{cov_name}"] * cov_val_t
                    
                    p_t[1] += self.true_betas[f"beta_p2_{cov_name}"] * cov_val_t
                    q_t[1] += self.true_betas[f"beta_q2_{cov_name}"] * cov_val_t
                    m_t[1] += self.true_betas[f"beta_m2_{cov_name}"] * cov_val_t

                    # Find index for alpha_1_2 and alpha_2_1 in alpha_base_flat
                    alpha_1_2_idx = -1
                    alpha_2_1_idx = -1
                    current_alpha_idx = 0
                    for i_prod in range(num_products):
                        for j_prod in range(num_products):
                            if i_prod != j_prod:
                                if i_prod == 0 and j_prod == 1:
                                    alpha_1_2_idx = current_alpha_idx
                                if i_prod == 1 and j_prod == 0:
                                    alpha_2_1_idx = current_alpha_idx
                                current_alpha_idx += 1
                    
                    if alpha_1_2_idx != -1:
                        alpha_t_flat[alpha_1_2_idx] += self.true_betas[f"beta_alpha_1_2_{cov_name}"] * cov_val_t
                    if alpha_2_1_idx != -1:
                        alpha_t_flat[alpha_2_1_idx] += self.true_betas[f"beta_alpha_2_1_{cov_name}"] * cov_val_t

            alpha_t = np.zeros((num_products, num_products))
            alpha_idx = 0
            for i_prod in range(num_products):
                for j_prod in range(num_products):
                    if i_prod != j_prod:
                        alpha_t[i_prod, j_prod] = alpha_t_flat[alpha_idx]
                        alpha_idx += 1

            dydt = np.zeros_like(y_val)
            for i_prod in range(num_products):
                interaction_term = sum(alpha_t[i_prod, j_prod] * y_val[j_prod] for j_prod in range(num_products) if i_prod != j_prod)
                dydt[i_prod] = (p_t[i_prod] + q_t[i_prod] * y_val[i_prod] / m_t[i_prod]) * (m_t[i_prod] - y_val[i_prod] - interaction_term) if m_t[i_prod] > 0 else 0
            return dydt

    # Initialize data generation model with base parameters
    data_gen_model = DataGenModel(n_products=2, covariates=['cov1'])
    data_gen_model.params_ = {
        "p1": p1_true_base, "q1": q1_true_base, "m1": m1_true_base,
        "p2": p2_true_base, "q2": q2_true_base, "m2": m2_true_base,
        "alpha_1_2": alpha12_true_base, "alpha_2_1": alpha21_true_base,
        # True betas are handled by the overridden differential_equation
        "beta_p1_cov1": 0.0, "beta_q1_cov1": 0.0, "beta_m1_cov1": 0.0, "beta_alpha_1_2_cov1": 0.0,
        "beta_p2_cov1": 0.0, "beta_q2_cov1": 0.0, "beta_m2_cov1": 0.0, "beta_alpha_2_1_cov1": 0.0,
    }

    # Generate data
    y_true = data_gen_model.predict(t, y0=np.array([1e-6, 1e-6]), covariates={'cov1': cov_values})
    np.random.seed(43)
    y_obs = np.maximum.accumulate(y_true + np.random.normal(0, 10, y_true.shape))
    y_obs[y_obs < 0] = 0

    # Fit the model with covariates
    model = MultiProductDiffusionModel(n_products=2, covariates=['cov1'])
    model.fit(t, y_obs, covariates={'cov1': cov_values})

    assert model.params_ is not None
    assert len(model.params_) == len(model.param_names)

    # Check if beta coefficients are reasonably close to true values (allowing for fitting noise)
    assert np.isclose(model.params_["beta_p1_cov1"], beta_p1_cov1_true, atol=0.01)
    assert np.isclose(model.params_["beta_q1_cov1"], beta_q1_cov1_true, atol=0.05)
    assert np.isclose(model.params_["beta_m1_cov1"], beta_m1_cov1_true, atol=50) # m can be harder to fit precisely
    assert np.isclose(model.params_["beta_alpha_1_2_cov1"], beta_alpha12_cov1_true, atol=0.05)

    assert np.isclose(model.params_["beta_p2_cov1"], beta_p2_cov1_true, atol=0.01)
    assert np.isclose(model.params_["beta_q2_cov1"], beta_q2_cov1_true, atol=0.05)
    assert np.isclose(model.params_["beta_m2_cov1"], beta_m2_cov1_true, atol=50)
    assert np.isclose(model.params_["beta_alpha_2_1_cov1"], beta_alpha21_cov1_true, atol=0.05)

    predictions = model.predict(t, y_obs[0, :], covariates={'cov1': cov_values})
    assert predictions.shape == y_obs.shape
    assert np.all(predictions >= 0)

    score = model.score(t, y_obs, y_obs[0, :], covariates={'cov1': cov_values})
    assert score > 0.7 # Should be a good fit with covariates

    # Test predict_adoption_rate with covariates
    rates = model.predict_adoption_rate(t, y_obs[0, :], covariates={'cov1': cov_values})
    assert rates.shape == y_obs.shape
    assert np.all(rates >= -1e-6) # Rates can be very slightly negative due to noise/fitting, but should be near 0 or positive

def test_multi_product_model_predict_basic():
    """Test the predict method of the MultiProductDiffusionModel with basic parameters."""
    p_vals = [0.02, 0.015]
    Q_matrix = [[0.3, 0.05], [0.03, 0.25]]
    m_vals = [1000, 800]
    product_names = ["ProdA", "ProdB"]

    model = MultiProductDiffusionModel(n_products=2, p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)
    
    t = np.arange(1, 101) # Use a longer time horizon for more meaningful prediction
    predictions_df = model.predict(t, y0=np.array([1e-6, 1e-6]))
    
    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) == len(t)
    assert list(predictions_df.columns) == product_names
    assert np.all(predictions_df.values >= 0)
    # Check if cumulative (each product's adoption should be non-decreasing)
    for col in product_names:
        assert np.all(np.diff(predictions_df[col].values) >= -1e-6)