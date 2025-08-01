"""
Tests for the BlackJaxFitter.
"""
from scipy import stats
import jax.numpy as jnp
import pytest

from innovate.diffuse.bass import BassModel
from innovate.fitters.blackjax_fitter import BlackJaxFitter


@pytest.mark.fitter
def test_blackjax_fitter():
    """
    Tests the BlackJaxFitter with the BassModel.
    """
    model = BassModel()
    fitter = BlackJaxFitter(model)

    # Generate some synthetic data
    t = jnp.arange(20)
    y = model.predict(t, p=0.03, q=0.38, m=1000)

    def log_probability_fn(params):
        p = params["p"]
        q = params["q"]
        m = params["m"]

        # Priors
        log_prior = (
            stats.uniform.logpdf(p, 0, 1)
            + stats.uniform.logpdf(q, 0, 1)
            + stats.norm.logpdf(m, 1000, 200)
        )

        # Likelihood
        y_pred = model.predict(t, **params)
        log_likelihood = jnp.sum(stats.norm.logpdf(y, y_pred, 1.0))

        return log_prior + log_likelihood

    initial_params = {"p": 0.5, "q": 0.5, "m": 1000.0}

    # Fit the model to the data
    fitter.fit(y, log_probability_fn, initial_params)

    # Check that the parameter estimates are reasonable
    estimates = fitter.get_parameter_estimates()
    assert "p" in estimates
    assert "q" in estimates
    assert "m" in estimates
    assert 0.0 < estimates["p"] < 1.0
    assert 0.0 < estimates["q"] < 1.0
    assert 800 < estimates["m"] < 1200
