"""
A robust Bayesian fitter using BlackJAX for MCMC sampling.

This implementation replaces the problematic PyMC-based BayesianFitter 
that suffered from segmentation faults. BlackJAX provides a more stable
JAX-based alternative for Bayesian inference.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
import warnings

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class BayesianFitter:
    """
    A Bayesian fitter using BlackJAX for robust parameter estimation.
    
    This fitter uses Hamiltonian Monte Carlo (HMC) via the NUTS sampler
    to perform Bayesian inference on diffusion model parameters. It provides
    uncertainty quantification and robust parameter estimates.
    
    Parameters
    ----------
    num_chains : int, default=4
        Number of MCMC chains to run in parallel
    num_warmup : int, default=1000  
        Number of warmup/burn-in steps per chain
    num_samples : int, default=1000
        Number of samples to draw per chain after warmup
    step_size : float, optional
        Initial step size for the sampler (auto-tuned if None)
    target_accept_rate : float, default=0.8
        Target acceptance rate for step size adaptation
    max_tree_depth : int, default=10
        Maximum tree depth for NUTS sampler
    """
    
    def __init__(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: Optional[float] = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.step_size = step_size
        self.target_accept_rate = target_accept_rate
        self.max_tree_depth = max_tree_depth
        self.random_seed = random_seed
        
        # State storage
        self.mcmc_results_ = None
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None
        
    def fit(
        self, 
        model: Any, 
        t: Union[np.ndarray, List, Sequence], 
        y: Union[np.ndarray, List, Sequence],
        **kwargs
    ) -> 'BayesianFitter':
        """
        Fit the model using Bayesian inference.
        
        Parameters
        ----------
        model : Model
            The diffusion model to fit (e.g., BassModel, LogisticModel)
        t : array-like
            Time points
        y : array-like  
            Observed adoption/cumulative values
        **kwargs
            Additional arguments (currently unused)
            
        Returns
        -------
        self : BayesianFitter
            Returns self for method chaining
        """
        # Store model and data
        self.model_ = model
        self.data_ = (np.asarray(t), np.asarray(y))
        
        # Convert to JAX arrays
        t_jax = jnp.asarray(t)
        y_jax = jnp.asarray(y)
        
        # Define log probability function
        log_prob_fn = self._create_log_probability_function(model, t_jax, y_jax)
        
        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)
        
        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)
        
        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()
        
        return self
    
    def _create_log_probability_function(
        self, 
        model: Any, 
        t: jnp.ndarray, 
        y: jnp.ndarray
    ) -> Callable:
        """Create log probability function for the model."""
        
        def log_prob(params_dict: Dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf
                
                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)
                
                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(
                    -0.5 * ((y - predictions) / sigma) ** 2 
                    - 0.5 * jnp.log(2 * jnp.pi * sigma**2)
                )
                
                return log_prior + log_likelihood
                
            except Exception:
                return -jnp.inf
        
        return log_prob
    
    def _compute_log_prior(
        self, 
        params_dict: Dict[str, float], 
        model: Any, 
        t: jnp.ndarray, 
        y: jnp.ndarray
    ) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0
            
            for param_name, value in params_dict.items():
                if param_name in bounds:
                    lower, upper = bounds[param_name]
                    
                    # Convert None bounds to reasonable values
                    if lower is None:
                        lower = -1e6
                    if upper is None:
                        upper = 1e6
                    
                    # Check bounds
                    if value < lower or value > upper:
                        return -jnp.inf
                    
                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)
            
            return log_prior
            
        except Exception:
            return -jnp.inf
    
    def _model_predict(
        self, 
        model: Any, 
        t: jnp.ndarray, 
        params_dict: Dict[str, float]
    ) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, 'params_', None)
        model.params_ = params_dict
        
        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params
    
    def _get_initial_parameters(
        self, 
        model: Any, 
        t: Union[np.ndarray, List], 
        y: Union[np.ndarray, List]
    ) -> Dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, 'initial_guesses'):
            return model.initial_guesses(t, y)
        
        # Fallback for models without initial_guesses
        param_names = getattr(model, 'param_names', ['p', 'q', 'm'])
        return {name: 0.1 for name in param_names}
    
    def _run_mcmc(
        self, 
        log_prob_fn: Callable, 
        initial_params: Dict[str, float]
    ) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)
        
        # Convert parameter dict to array for sampling
        param_names = list(initial_params.keys())
        initial_position = jnp.array([initial_params[name] for name in param_names])
        
        # Create log prob function that takes array input
        def log_prob_array(position_array):
            params_dict = {name: position_array[i] for i, name in enumerate(param_names)}
            return log_prob_fn(params_dict)
        
        try:
            # Adaptation phase
            warmup = blackjax.window_adaptation(
                blackjax.nuts, 
                log_prob_array,
                target_acceptance_rate=self.target_accept_rate,
            )
            
            (state, parameters), _ = warmup.run(
                rng_key, 
                initial_position, 
                num_steps=self.num_warmup
            )
            
            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)
            
            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position
            
            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)
                
                (final_state, _), chain_samples = jax.lax.scan(
                    one_step, 
                    (state, None), 
                    sample_keys
                )
                all_samples.append(chain_samples)
            
            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)
            
            # Convert back to parameter dictionaries
            self.posterior_samples_ = {
                param_names[i]: samples_array[:, :, i] 
                for i in range(len(param_names))
            }
            
            self.mcmc_results_ = {
                'samples': samples_array,
                'param_names': param_names,
                'final_state': final_state,
            }
            
        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {str(e)}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value)
                for name, value in initial_params.items()
            }
    
    def get_parameter_estimates(self) -> Dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        return {
            param: float(jnp.mean(samples))
            for param, samples in self.posterior_samples_.items()
        }
    
    def get_confidence_intervals(
        self, 
        credible_mass: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)
        
        return intervals
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                'mean': float(jnp.mean(flat_samples)),
                'std': float(jnp.std(flat_samples)),
                'median': float(jnp.median(flat_samples)),
                '2.5%': float(jnp.percentile(flat_samples, 2.5)),
                '97.5%': float(jnp.percentile(flat_samples, 97.5)),
                'n_eff': float(len(flat_samples)),  # Simplified
            }
        
        return summary
    
    def plot_trace(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)
    
    def plot_posterior(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)
    
    def _to_inference_data(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_,
            coords={'chain': range(self.num_chains), 'draw': range(self.num_samples)}
        )