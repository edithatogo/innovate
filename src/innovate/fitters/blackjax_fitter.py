"""A Bayesian fitter that uses the BlackJAX library for MCMC sampling.
"""
from typing import Any, Callable, Dict, Tuple

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
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
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
        self.states = None
        self.kernel = blackjax.nuts

    def fit(
        self,
        y: jnp.ndarray,
        log_probability_fn: Callable,
        initial_params: Dict[str, float],
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)
            return states

        all_states = []
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
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def get_parameter_estimates(self) -> Dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {
            param: jnp.concatenate([p[param] for p in positions])
            for param in positions[0]
        }

        estimates = {
            param: float(jnp.mean(samples)) for param, samples in param_samples.items()
        }
        return estimates

    def _get_inference_data(self):
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states])
            for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
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
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        return az.summary(idata)

    def plot_trace(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def plot_posterior(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)
