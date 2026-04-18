"""A Bayesian fitter that uses the BlackJAX library for MCMC sampling."""

from collections.abc import Callable
from typing import Any

import arviz as az
import blackjax
import jax
import jax.numpy as jnp


class BlackJaxFitter:
    """A fitter that uses BlackJAX for Bayesian parameter estimation.

    This fitter provides a flexible way to perform Bayesian inference by
    leveraging the high-performance samplers in BlackJAX. It is designed
    to be used within the JAX ecosystem and is suitable for models that
    can be expressed as a log-probability function.
    """

    def __init__(
        self,
        model: Any,
        num_chains: int = 1,
        num_warmup: int = 100,
        num_samples: int = 100,
    ):
        """Initializes the BlackJaxFitter.

        Args:
        ----
            model: The model to fit.
            num_chains: The number of chains to run.
            num_warmup: The number of warmup steps for the sampler.
            num_samples: The number of samples to draw from the posterior.
        """
        self.model = model
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.samples = None
        self.param_names = None

    def fit(
        self,
        y: jnp.ndarray,
        log_probability_fn: Callable,
        initial_params: dict[str, float],
        **kwargs: Any,
    ) -> None:
        """Fits the model to the data using a BlackJAX sampler.

        Args:
        ----
            y: The observed data.
            log_probability_fn: A function that takes a dictionary of
                parameters and returns the log-probability of the model.
            initial_params: A dictionary of starting values for the parameters.
            **kwargs: Additional arguments to pass to the inference loop.
        """
        rng_key = jax.random.PRNGKey(0)

        param_names = list(initial_params.keys())
        self.param_names = param_names

        def inference_loop(rng_key, sampler, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                new_state, info = sampler.step(rng_key, state)
                return new_state, new_state.position

            keys = jax.random.split(rng_key, num_samples)
            _, samples = jax.lax.scan(one_step, initial_state, keys)
            return samples

        all_samples = []
        for i in range(self.num_chains):
            chain_rng_key, rng_key = jax.random.split(rng_key)

            # Adapt step size and mass matrix
            adapt = blackjax.window_adaptation(blackjax.nuts, log_probability_fn)
            (last_state, parameters), _ = adapt.run(
                chain_rng_key,
                initial_params,
                self.num_warmup,
            )

            # Sample from the posterior
            sampler = blackjax.nuts(log_probability_fn, **parameters)
            samples = inference_loop(chain_rng_key, sampler, last_state, self.num_samples)
            all_samples.append(samples)

        self.samples = all_samples

    def get_parameter_estimates(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.samples is None or self.param_names is None:
            raise RuntimeError("The model has not been fitted yet.")

        param_samples = {
            name: jnp.concatenate([chain_samples[name] for chain_samples in self.samples]) for name in self.param_names
        }

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def _get_inference_data(self):
        if self.samples is None or self.param_names is None:
            raise RuntimeError("The model has not been fitted yet.")

        posterior_samples = {
            param: jnp.stack([chain_samples[param] for chain_samples in self.samples]) for param in self.param_names
        }

        return az.from_dict(posterior_samples)

    def get_confidence_intervals(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.samples is None or self.param_names is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def get_summary(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.samples is None or self.param_names is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        return az.summary(idata)

    def plot_trace(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.samples is None or self.param_names is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def plot_posterior(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)
