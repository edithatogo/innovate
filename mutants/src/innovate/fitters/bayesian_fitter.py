"""
A robust Bayesian fitter using BlackJAX for MCMC sampling.

This implementation replaces the problematic PyMC-based BayesianFitter
that suffered from segmentation faults. BlackJAX provides a more stable
JAX-based alternative for Bayesian inference.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


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
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        args = [num_chains, num_warmup, num_samples, step_size, target_accept_rate, max_tree_depth, random_seed]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ__init____mutmut_orig(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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

    def xǁBayesianFitterǁ__init____mutmut_1(
        self,
        num_chains: int = 5,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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

    def xǁBayesianFitterǁ__init____mutmut_2(
        self,
        num_chains: int = 4,
        num_warmup: int = 1001,
        num_samples: int = 1000,
        step_size: float | None = None,
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

    def xǁBayesianFitterǁ__init____mutmut_3(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1001,
        step_size: float | None = None,
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

    def xǁBayesianFitterǁ__init____mutmut_4(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 1.8,
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

    def xǁBayesianFitterǁ__init____mutmut_5(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 11,
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

    def xǁBayesianFitterǁ__init____mutmut_6(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 43,
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

    def xǁBayesianFitterǁ__init____mutmut_7(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = None
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

    def xǁBayesianFitterǁ__init____mutmut_8(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = num_chains
        self.num_warmup = None
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

    def xǁBayesianFitterǁ__init____mutmut_9(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = None
        self.step_size = step_size
        self.target_accept_rate = target_accept_rate
        self.max_tree_depth = max_tree_depth
        self.random_seed = random_seed

        # State storage
        self.mcmc_results_ = None
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_10(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.step_size = None
        self.target_accept_rate = target_accept_rate
        self.max_tree_depth = max_tree_depth
        self.random_seed = random_seed

        # State storage
        self.mcmc_results_ = None
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_11(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.step_size = step_size
        self.target_accept_rate = None
        self.max_tree_depth = max_tree_depth
        self.random_seed = random_seed

        # State storage
        self.mcmc_results_ = None
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_12(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        random_seed: int = 42,
    ):
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.step_size = step_size
        self.target_accept_rate = target_accept_rate
        self.max_tree_depth = None
        self.random_seed = random_seed

        # State storage
        self.mcmc_results_ = None
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_13(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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
        self.random_seed = None

        # State storage
        self.mcmc_results_ = None
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_14(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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
        self.mcmc_results_ = ""
        self.posterior_samples_ = None
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_15(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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
        self.posterior_samples_ = ""
        self.model_ = None
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_16(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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
        self.model_ = ""
        self.data_ = None

    def xǁBayesianFitterǁ__init____mutmut_17(
        self,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        step_size: float | None = None,
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
        self.data_ = ""
    
    xǁBayesianFitterǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ__init____mutmut_1': xǁBayesianFitterǁ__init____mutmut_1, 
        'xǁBayesianFitterǁ__init____mutmut_2': xǁBayesianFitterǁ__init____mutmut_2, 
        'xǁBayesianFitterǁ__init____mutmut_3': xǁBayesianFitterǁ__init____mutmut_3, 
        'xǁBayesianFitterǁ__init____mutmut_4': xǁBayesianFitterǁ__init____mutmut_4, 
        'xǁBayesianFitterǁ__init____mutmut_5': xǁBayesianFitterǁ__init____mutmut_5, 
        'xǁBayesianFitterǁ__init____mutmut_6': xǁBayesianFitterǁ__init____mutmut_6, 
        'xǁBayesianFitterǁ__init____mutmut_7': xǁBayesianFitterǁ__init____mutmut_7, 
        'xǁBayesianFitterǁ__init____mutmut_8': xǁBayesianFitterǁ__init____mutmut_8, 
        'xǁBayesianFitterǁ__init____mutmut_9': xǁBayesianFitterǁ__init____mutmut_9, 
        'xǁBayesianFitterǁ__init____mutmut_10': xǁBayesianFitterǁ__init____mutmut_10, 
        'xǁBayesianFitterǁ__init____mutmut_11': xǁBayesianFitterǁ__init____mutmut_11, 
        'xǁBayesianFitterǁ__init____mutmut_12': xǁBayesianFitterǁ__init____mutmut_12, 
        'xǁBayesianFitterǁ__init____mutmut_13': xǁBayesianFitterǁ__init____mutmut_13, 
        'xǁBayesianFitterǁ__init____mutmut_14': xǁBayesianFitterǁ__init____mutmut_14, 
        'xǁBayesianFitterǁ__init____mutmut_15': xǁBayesianFitterǁ__init____mutmut_15, 
        'xǁBayesianFitterǁ__init____mutmut_16': xǁBayesianFitterǁ__init____mutmut_16, 
        'xǁBayesianFitterǁ__init____mutmut_17': xǁBayesianFitterǁ__init____mutmut_17
    }
    xǁBayesianFitterǁ__init____mutmut_orig.__name__ = 'xǁBayesianFitterǁ__init__'

    def fit(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
        args = [model, t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁfit__mutmut_orig(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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

    def xǁBayesianFitterǁfit__mutmut_1(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self.model_ = None
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

    def xǁBayesianFitterǁfit__mutmut_2(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self.data_ = None

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

    def xǁBayesianFitterǁfit__mutmut_3(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self.data_ = (np.asarray(None), np.asarray(y))

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

    def xǁBayesianFitterǁfit__mutmut_4(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self.data_ = (np.asarray(t), np.asarray(None))

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

    def xǁBayesianFitterǁfit__mutmut_5(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        t_jax = None
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

    def xǁBayesianFitterǁfit__mutmut_6(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        t_jax = jnp.asarray(None)
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

    def xǁBayesianFitterǁfit__mutmut_7(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        y_jax = None

        # Define log probability function
        log_prob_fn = self._create_log_probability_function(model, t_jax, y_jax)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_8(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        y_jax = jnp.asarray(None)

        # Define log probability function
        log_prob_fn = self._create_log_probability_function(model, t_jax, y_jax)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_9(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = None

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_10(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = self._create_log_probability_function(None, t_jax, y_jax)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_11(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = self._create_log_probability_function(model, None, y_jax)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_12(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = self._create_log_probability_function(model, t_jax, None)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_13(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = self._create_log_probability_function(t_jax, y_jax)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_14(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = self._create_log_probability_function(model, y_jax)

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_15(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        log_prob_fn = self._create_log_probability_function(model, t_jax, )

        # Get initial parameter values and bounds
        initial_params = self._get_initial_parameters(model, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_16(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = None

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_17(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = self._get_initial_parameters(None, t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_18(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = self._get_initial_parameters(model, None, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_19(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = self._get_initial_parameters(model, t, None)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_20(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = self._get_initial_parameters(t, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_21(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = self._get_initial_parameters(model, y)

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_22(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        initial_params = self._get_initial_parameters(model, t, )

        # Run MCMC sampling
        self._run_mcmc(log_prob_fn, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_23(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self._run_mcmc(None, initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_24(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self._run_mcmc(log_prob_fn, None)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_25(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self._run_mcmc(initial_params)

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_26(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        self._run_mcmc(log_prob_fn, )

        # Set fitted parameters to posterior means
        model.params_ = self.get_parameter_estimates()

        return self

    def xǁBayesianFitterǁfit__mutmut_27(
        self, model: Any, t: np.ndarray | list | Sequence, y: np.ndarray | list | Sequence, **kwargs
    ) -> "BayesianFitter":
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
        model.params_ = None

        return self
    
    xǁBayesianFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁfit__mutmut_1': xǁBayesianFitterǁfit__mutmut_1, 
        'xǁBayesianFitterǁfit__mutmut_2': xǁBayesianFitterǁfit__mutmut_2, 
        'xǁBayesianFitterǁfit__mutmut_3': xǁBayesianFitterǁfit__mutmut_3, 
        'xǁBayesianFitterǁfit__mutmut_4': xǁBayesianFitterǁfit__mutmut_4, 
        'xǁBayesianFitterǁfit__mutmut_5': xǁBayesianFitterǁfit__mutmut_5, 
        'xǁBayesianFitterǁfit__mutmut_6': xǁBayesianFitterǁfit__mutmut_6, 
        'xǁBayesianFitterǁfit__mutmut_7': xǁBayesianFitterǁfit__mutmut_7, 
        'xǁBayesianFitterǁfit__mutmut_8': xǁBayesianFitterǁfit__mutmut_8, 
        'xǁBayesianFitterǁfit__mutmut_9': xǁBayesianFitterǁfit__mutmut_9, 
        'xǁBayesianFitterǁfit__mutmut_10': xǁBayesianFitterǁfit__mutmut_10, 
        'xǁBayesianFitterǁfit__mutmut_11': xǁBayesianFitterǁfit__mutmut_11, 
        'xǁBayesianFitterǁfit__mutmut_12': xǁBayesianFitterǁfit__mutmut_12, 
        'xǁBayesianFitterǁfit__mutmut_13': xǁBayesianFitterǁfit__mutmut_13, 
        'xǁBayesianFitterǁfit__mutmut_14': xǁBayesianFitterǁfit__mutmut_14, 
        'xǁBayesianFitterǁfit__mutmut_15': xǁBayesianFitterǁfit__mutmut_15, 
        'xǁBayesianFitterǁfit__mutmut_16': xǁBayesianFitterǁfit__mutmut_16, 
        'xǁBayesianFitterǁfit__mutmut_17': xǁBayesianFitterǁfit__mutmut_17, 
        'xǁBayesianFitterǁfit__mutmut_18': xǁBayesianFitterǁfit__mutmut_18, 
        'xǁBayesianFitterǁfit__mutmut_19': xǁBayesianFitterǁfit__mutmut_19, 
        'xǁBayesianFitterǁfit__mutmut_20': xǁBayesianFitterǁfit__mutmut_20, 
        'xǁBayesianFitterǁfit__mutmut_21': xǁBayesianFitterǁfit__mutmut_21, 
        'xǁBayesianFitterǁfit__mutmut_22': xǁBayesianFitterǁfit__mutmut_22, 
        'xǁBayesianFitterǁfit__mutmut_23': xǁBayesianFitterǁfit__mutmut_23, 
        'xǁBayesianFitterǁfit__mutmut_24': xǁBayesianFitterǁfit__mutmut_24, 
        'xǁBayesianFitterǁfit__mutmut_25': xǁBayesianFitterǁfit__mutmut_25, 
        'xǁBayesianFitterǁfit__mutmut_26': xǁBayesianFitterǁfit__mutmut_26, 
        'xǁBayesianFitterǁfit__mutmut_27': xǁBayesianFitterǁfit__mutmut_27
    }
    xǁBayesianFitterǁfit__mutmut_orig.__name__ = 'xǁBayesianFitterǁfit'

    def _create_log_probability_function(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        args = [model, t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ_create_log_probability_function__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ_create_log_probability_function__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_orig(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_1(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = None
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_2(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(None, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_3(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, None, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_4(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, None, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_5(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, None)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_6(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_7(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_8(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_9(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, )
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_10(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_11(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(None):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_12(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return +jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_13(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = None

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_14(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(None, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_15(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, None, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_16(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, None)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_17(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_18(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_19(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, )

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_20(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = None  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_21(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(None, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_22(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, None)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_23(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_24(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, )  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_25(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(1.01, jnp.std(y) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_26(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) / 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_27(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(None) * 0.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_28(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
            """Log probability function for MCMC sampling."""
            try:
                # Apply parameter bounds as priors
                log_prior = self._compute_log_prior(params_dict, model, t, y)
                if not jnp.isfinite(log_prior):
                    return -jnp.inf

                # Compute model predictions
                predictions = self._model_predict(model, t, params_dict)

                # Compute log likelihood (assuming Gaussian noise)
                sigma = jnp.maximum(0.01, jnp.std(y) * 1.1)  # Minimum noise level
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_29(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = None

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_30(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(None)

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_31(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 + 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_32(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 / ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_33(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(+0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_34(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-1.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_35(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) * 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_36(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) * sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_37(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y + predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_38(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 3 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_39(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 / jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_40(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 1.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_41(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(None))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_42(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi / sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_43(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 / jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_44(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(3 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_45(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma * 2))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_46(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**3))

                return log_prior + log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_47(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior - log_likelihood

            except Exception:
                return -jnp.inf

        return log_prob

    def xǁBayesianFitterǁ_create_log_probability_function__mutmut_48(self, model: Any, t: jnp.ndarray, y: jnp.ndarray) -> Callable:
        """Create log probability function for the model."""

        def log_prob(params_dict: dict[str, float]) -> float:
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
                log_likelihood = jnp.sum(-0.5 * ((y - predictions) / sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2))

                return log_prior + log_likelihood

            except Exception:
                return +jnp.inf

        return log_prob
    
    xǁBayesianFitterǁ_create_log_probability_function__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ_create_log_probability_function__mutmut_1': xǁBayesianFitterǁ_create_log_probability_function__mutmut_1, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_2': xǁBayesianFitterǁ_create_log_probability_function__mutmut_2, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_3': xǁBayesianFitterǁ_create_log_probability_function__mutmut_3, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_4': xǁBayesianFitterǁ_create_log_probability_function__mutmut_4, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_5': xǁBayesianFitterǁ_create_log_probability_function__mutmut_5, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_6': xǁBayesianFitterǁ_create_log_probability_function__mutmut_6, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_7': xǁBayesianFitterǁ_create_log_probability_function__mutmut_7, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_8': xǁBayesianFitterǁ_create_log_probability_function__mutmut_8, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_9': xǁBayesianFitterǁ_create_log_probability_function__mutmut_9, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_10': xǁBayesianFitterǁ_create_log_probability_function__mutmut_10, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_11': xǁBayesianFitterǁ_create_log_probability_function__mutmut_11, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_12': xǁBayesianFitterǁ_create_log_probability_function__mutmut_12, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_13': xǁBayesianFitterǁ_create_log_probability_function__mutmut_13, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_14': xǁBayesianFitterǁ_create_log_probability_function__mutmut_14, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_15': xǁBayesianFitterǁ_create_log_probability_function__mutmut_15, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_16': xǁBayesianFitterǁ_create_log_probability_function__mutmut_16, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_17': xǁBayesianFitterǁ_create_log_probability_function__mutmut_17, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_18': xǁBayesianFitterǁ_create_log_probability_function__mutmut_18, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_19': xǁBayesianFitterǁ_create_log_probability_function__mutmut_19, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_20': xǁBayesianFitterǁ_create_log_probability_function__mutmut_20, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_21': xǁBayesianFitterǁ_create_log_probability_function__mutmut_21, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_22': xǁBayesianFitterǁ_create_log_probability_function__mutmut_22, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_23': xǁBayesianFitterǁ_create_log_probability_function__mutmut_23, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_24': xǁBayesianFitterǁ_create_log_probability_function__mutmut_24, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_25': xǁBayesianFitterǁ_create_log_probability_function__mutmut_25, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_26': xǁBayesianFitterǁ_create_log_probability_function__mutmut_26, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_27': xǁBayesianFitterǁ_create_log_probability_function__mutmut_27, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_28': xǁBayesianFitterǁ_create_log_probability_function__mutmut_28, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_29': xǁBayesianFitterǁ_create_log_probability_function__mutmut_29, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_30': xǁBayesianFitterǁ_create_log_probability_function__mutmut_30, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_31': xǁBayesianFitterǁ_create_log_probability_function__mutmut_31, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_32': xǁBayesianFitterǁ_create_log_probability_function__mutmut_32, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_33': xǁBayesianFitterǁ_create_log_probability_function__mutmut_33, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_34': xǁBayesianFitterǁ_create_log_probability_function__mutmut_34, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_35': xǁBayesianFitterǁ_create_log_probability_function__mutmut_35, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_36': xǁBayesianFitterǁ_create_log_probability_function__mutmut_36, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_37': xǁBayesianFitterǁ_create_log_probability_function__mutmut_37, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_38': xǁBayesianFitterǁ_create_log_probability_function__mutmut_38, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_39': xǁBayesianFitterǁ_create_log_probability_function__mutmut_39, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_40': xǁBayesianFitterǁ_create_log_probability_function__mutmut_40, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_41': xǁBayesianFitterǁ_create_log_probability_function__mutmut_41, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_42': xǁBayesianFitterǁ_create_log_probability_function__mutmut_42, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_43': xǁBayesianFitterǁ_create_log_probability_function__mutmut_43, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_44': xǁBayesianFitterǁ_create_log_probability_function__mutmut_44, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_45': xǁBayesianFitterǁ_create_log_probability_function__mutmut_45, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_46': xǁBayesianFitterǁ_create_log_probability_function__mutmut_46, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_47': xǁBayesianFitterǁ_create_log_probability_function__mutmut_47, 
        'xǁBayesianFitterǁ_create_log_probability_function__mutmut_48': xǁBayesianFitterǁ_create_log_probability_function__mutmut_48
    }
    xǁBayesianFitterǁ_create_log_probability_function__mutmut_orig.__name__ = 'xǁBayesianFitterǁ_create_log_probability_function'

    def _compute_log_prior(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        args = [params_dict, model, t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ_compute_log_prior__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ_compute_log_prior__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_orig(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_1(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = None
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_2(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(None, y)
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_3(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, None)
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_4(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(y)
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_5(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, )
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_6(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = None

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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_7(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 1.0

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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_8(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0

            for param_name, value in params_dict.items():
                if param_name not in bounds:
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_9(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0

            for param_name, value in params_dict.items():
                if param_name in bounds:
                    lower, upper = None

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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_10(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0

            for param_name, value in params_dict.items():
                if param_name in bounds:
                    lower, upper = bounds[param_name]

                    # Convert None bounds to reasonable values
                    if lower is not None:
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_11(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0

            for param_name, value in params_dict.items():
                if param_name in bounds:
                    lower, upper = bounds[param_name]

                    # Convert None bounds to reasonable values
                    if lower is None:
                        lower = None
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_12(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0

            for param_name, value in params_dict.items():
                if param_name in bounds:
                    lower, upper = bounds[param_name]

                    # Convert None bounds to reasonable values
                    if lower is None:
                        lower = +1e6
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_13(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log prior probability based on parameter bounds."""
        try:
            bounds = model.bounds(t, y)
            log_prior = 0.0

            for param_name, value in params_dict.items():
                if param_name in bounds:
                    lower, upper = bounds[param_name]

                    # Convert None bounds to reasonable values
                    if lower is None:
                        lower = -1000001.0
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_14(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if upper is not None:
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

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_15(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        upper = None

                    # Check bounds
                    if value < lower or value > upper:
                        return -jnp.inf

                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_16(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        upper = 1000001.0

                    # Check bounds
                    if value < lower or value > upper:
                        return -jnp.inf

                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_17(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if value < lower and value > upper:
                        return -jnp.inf

                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_18(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if value <= lower or value > upper:
                        return -jnp.inf

                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_19(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if value < lower or value >= upper:
                        return -jnp.inf

                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_20(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        return +jnp.inf

                    # Uniform prior within bounds
                    if jnp.isfinite(upper) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_21(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if jnp.isfinite(upper) or jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_22(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if jnp.isfinite(None) and jnp.isfinite(lower):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_23(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                    if jnp.isfinite(upper) and jnp.isfinite(None):
                        log_prior -= jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_24(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        log_prior = jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_25(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        log_prior += jnp.log(upper - lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_26(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        log_prior -= jnp.log(None)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_27(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
                        log_prior -= jnp.log(upper + lower)

            return log_prior

        except Exception:
            return -jnp.inf

    def xǁBayesianFitterǁ_compute_log_prior__mutmut_28(self, params_dict: dict[str, float], model: Any, t: jnp.ndarray, y: jnp.ndarray) -> float:
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
            return +jnp.inf
    
    xǁBayesianFitterǁ_compute_log_prior__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ_compute_log_prior__mutmut_1': xǁBayesianFitterǁ_compute_log_prior__mutmut_1, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_2': xǁBayesianFitterǁ_compute_log_prior__mutmut_2, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_3': xǁBayesianFitterǁ_compute_log_prior__mutmut_3, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_4': xǁBayesianFitterǁ_compute_log_prior__mutmut_4, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_5': xǁBayesianFitterǁ_compute_log_prior__mutmut_5, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_6': xǁBayesianFitterǁ_compute_log_prior__mutmut_6, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_7': xǁBayesianFitterǁ_compute_log_prior__mutmut_7, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_8': xǁBayesianFitterǁ_compute_log_prior__mutmut_8, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_9': xǁBayesianFitterǁ_compute_log_prior__mutmut_9, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_10': xǁBayesianFitterǁ_compute_log_prior__mutmut_10, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_11': xǁBayesianFitterǁ_compute_log_prior__mutmut_11, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_12': xǁBayesianFitterǁ_compute_log_prior__mutmut_12, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_13': xǁBayesianFitterǁ_compute_log_prior__mutmut_13, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_14': xǁBayesianFitterǁ_compute_log_prior__mutmut_14, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_15': xǁBayesianFitterǁ_compute_log_prior__mutmut_15, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_16': xǁBayesianFitterǁ_compute_log_prior__mutmut_16, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_17': xǁBayesianFitterǁ_compute_log_prior__mutmut_17, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_18': xǁBayesianFitterǁ_compute_log_prior__mutmut_18, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_19': xǁBayesianFitterǁ_compute_log_prior__mutmut_19, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_20': xǁBayesianFitterǁ_compute_log_prior__mutmut_20, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_21': xǁBayesianFitterǁ_compute_log_prior__mutmut_21, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_22': xǁBayesianFitterǁ_compute_log_prior__mutmut_22, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_23': xǁBayesianFitterǁ_compute_log_prior__mutmut_23, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_24': xǁBayesianFitterǁ_compute_log_prior__mutmut_24, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_25': xǁBayesianFitterǁ_compute_log_prior__mutmut_25, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_26': xǁBayesianFitterǁ_compute_log_prior__mutmut_26, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_27': xǁBayesianFitterǁ_compute_log_prior__mutmut_27, 
        'xǁBayesianFitterǁ_compute_log_prior__mutmut_28': xǁBayesianFitterǁ_compute_log_prior__mutmut_28
    }
    xǁBayesianFitterǁ_compute_log_prior__mutmut_orig.__name__ = 'xǁBayesianFitterǁ_compute_log_prior'

    def _model_predict(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        args = [model, t, params_dict]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ_model_predict__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ_model_predict__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ_model_predict__mutmut_orig(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_1(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = None
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_2(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(None, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_3(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, None, None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_4(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr("params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_5(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_6(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", )
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_7(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "XXparams_XX", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_8(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "PARAMS_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_9(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = None

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_10(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = None
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_11(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(None)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_12(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(None)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_13(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(None, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_14(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, None)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_15(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_16(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, )
        finally:
            # Restore original parameters
            model.params_ = original_params

    def xǁBayesianFitterǁ_model_predict__mutmut_17(self, model: Any, t: jnp.ndarray, params_dict: dict[str, float]) -> jnp.ndarray:
        """Make predictions with the model using given parameters."""
        # Temporarily set parameters
        original_params = getattr(model, "params_", None)
        model.params_ = params_dict

        try:
            predictions = model.predict(t)
            return jnp.asarray(predictions)
        except Exception:
            return jnp.full_like(t, jnp.nan)
        finally:
            # Restore original parameters
            model.params_ = None
    
    xǁBayesianFitterǁ_model_predict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ_model_predict__mutmut_1': xǁBayesianFitterǁ_model_predict__mutmut_1, 
        'xǁBayesianFitterǁ_model_predict__mutmut_2': xǁBayesianFitterǁ_model_predict__mutmut_2, 
        'xǁBayesianFitterǁ_model_predict__mutmut_3': xǁBayesianFitterǁ_model_predict__mutmut_3, 
        'xǁBayesianFitterǁ_model_predict__mutmut_4': xǁBayesianFitterǁ_model_predict__mutmut_4, 
        'xǁBayesianFitterǁ_model_predict__mutmut_5': xǁBayesianFitterǁ_model_predict__mutmut_5, 
        'xǁBayesianFitterǁ_model_predict__mutmut_6': xǁBayesianFitterǁ_model_predict__mutmut_6, 
        'xǁBayesianFitterǁ_model_predict__mutmut_7': xǁBayesianFitterǁ_model_predict__mutmut_7, 
        'xǁBayesianFitterǁ_model_predict__mutmut_8': xǁBayesianFitterǁ_model_predict__mutmut_8, 
        'xǁBayesianFitterǁ_model_predict__mutmut_9': xǁBayesianFitterǁ_model_predict__mutmut_9, 
        'xǁBayesianFitterǁ_model_predict__mutmut_10': xǁBayesianFitterǁ_model_predict__mutmut_10, 
        'xǁBayesianFitterǁ_model_predict__mutmut_11': xǁBayesianFitterǁ_model_predict__mutmut_11, 
        'xǁBayesianFitterǁ_model_predict__mutmut_12': xǁBayesianFitterǁ_model_predict__mutmut_12, 
        'xǁBayesianFitterǁ_model_predict__mutmut_13': xǁBayesianFitterǁ_model_predict__mutmut_13, 
        'xǁBayesianFitterǁ_model_predict__mutmut_14': xǁBayesianFitterǁ_model_predict__mutmut_14, 
        'xǁBayesianFitterǁ_model_predict__mutmut_15': xǁBayesianFitterǁ_model_predict__mutmut_15, 
        'xǁBayesianFitterǁ_model_predict__mutmut_16': xǁBayesianFitterǁ_model_predict__mutmut_16, 
        'xǁBayesianFitterǁ_model_predict__mutmut_17': xǁBayesianFitterǁ_model_predict__mutmut_17
    }
    xǁBayesianFitterǁ_model_predict__mutmut_orig.__name__ = 'xǁBayesianFitterǁ_model_predict'

    def _get_initial_parameters(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        args = [model, t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ_get_initial_parameters__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ_get_initial_parameters__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_orig(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_1(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(None, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_2(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, None):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_3(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr("initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_4(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, ):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_5(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "XXinitial_guessesXX"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_6(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "INITIAL_GUESSES"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_7(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(None, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_8(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, None)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_9(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_10(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, )

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_11(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = None
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_12(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(None, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_13(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, None, ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_14(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", None)
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_15(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr("param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_16(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_17(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", )
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_18(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "XXparam_namesXX", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_19(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "PARAM_NAMES", ["p", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_20(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["XXpXX", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_21(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["P", "q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_22(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "XXqXX", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_23(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "Q", "m"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_24(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "XXmXX"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_25(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "M"])
        return dict.fromkeys(param_names, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_26(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(None, 0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_27(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, None)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_28(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(0.1)

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_29(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, )

    def xǁBayesianFitterǁ_get_initial_parameters__mutmut_30(self, model: Any, t: np.ndarray | list, y: np.ndarray | list) -> dict[str, float]:
        """Get initial parameter values for MCMC."""
        if hasattr(model, "initial_guesses"):
            return model.initial_guesses(t, y)

        # Fallback for models without initial_guesses
        param_names = getattr(model, "param_names", ["p", "q", "m"])
        return dict.fromkeys(param_names, 1.1)
    
    xǁBayesianFitterǁ_get_initial_parameters__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ_get_initial_parameters__mutmut_1': xǁBayesianFitterǁ_get_initial_parameters__mutmut_1, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_2': xǁBayesianFitterǁ_get_initial_parameters__mutmut_2, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_3': xǁBayesianFitterǁ_get_initial_parameters__mutmut_3, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_4': xǁBayesianFitterǁ_get_initial_parameters__mutmut_4, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_5': xǁBayesianFitterǁ_get_initial_parameters__mutmut_5, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_6': xǁBayesianFitterǁ_get_initial_parameters__mutmut_6, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_7': xǁBayesianFitterǁ_get_initial_parameters__mutmut_7, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_8': xǁBayesianFitterǁ_get_initial_parameters__mutmut_8, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_9': xǁBayesianFitterǁ_get_initial_parameters__mutmut_9, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_10': xǁBayesianFitterǁ_get_initial_parameters__mutmut_10, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_11': xǁBayesianFitterǁ_get_initial_parameters__mutmut_11, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_12': xǁBayesianFitterǁ_get_initial_parameters__mutmut_12, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_13': xǁBayesianFitterǁ_get_initial_parameters__mutmut_13, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_14': xǁBayesianFitterǁ_get_initial_parameters__mutmut_14, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_15': xǁBayesianFitterǁ_get_initial_parameters__mutmut_15, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_16': xǁBayesianFitterǁ_get_initial_parameters__mutmut_16, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_17': xǁBayesianFitterǁ_get_initial_parameters__mutmut_17, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_18': xǁBayesianFitterǁ_get_initial_parameters__mutmut_18, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_19': xǁBayesianFitterǁ_get_initial_parameters__mutmut_19, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_20': xǁBayesianFitterǁ_get_initial_parameters__mutmut_20, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_21': xǁBayesianFitterǁ_get_initial_parameters__mutmut_21, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_22': xǁBayesianFitterǁ_get_initial_parameters__mutmut_22, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_23': xǁBayesianFitterǁ_get_initial_parameters__mutmut_23, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_24': xǁBayesianFitterǁ_get_initial_parameters__mutmut_24, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_25': xǁBayesianFitterǁ_get_initial_parameters__mutmut_25, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_26': xǁBayesianFitterǁ_get_initial_parameters__mutmut_26, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_27': xǁBayesianFitterǁ_get_initial_parameters__mutmut_27, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_28': xǁBayesianFitterǁ_get_initial_parameters__mutmut_28, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_29': xǁBayesianFitterǁ_get_initial_parameters__mutmut_29, 
        'xǁBayesianFitterǁ_get_initial_parameters__mutmut_30': xǁBayesianFitterǁ_get_initial_parameters__mutmut_30
    }
    xǁBayesianFitterǁ_get_initial_parameters__mutmut_orig.__name__ = 'xǁBayesianFitterǁ_get_initial_parameters'

    def _run_mcmc(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        args = [log_prob_fn, initial_params]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ_run_mcmc__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ_run_mcmc__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ_run_mcmc__mutmut_orig(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_1(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = None

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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_2(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(None)

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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_3(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = None
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_4(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = list(None)
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_5(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = list(initial_params.keys())
        initial_position = None

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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_6(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = list(initial_params.keys())
        initial_position = jnp.array(None)

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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_7(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = list(initial_params.keys())
        initial_position = jnp.array([initial_params[name] for name in param_names])

        # Create log prob function that takes array input
        def log_prob_array(position_array):
            params_dict = None
            return log_prob_fn(params_dict)

        try:
            # Adaptation phase
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                log_prob_array,
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_8(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = list(initial_params.keys())
        initial_position = jnp.array([initial_params[name] for name in param_names])

        # Create log prob function that takes array input
        def log_prob_array(position_array):
            params_dict = {name: position_array[i] for i, name in enumerate(None)}
            return log_prob_fn(params_dict)

        try:
            # Adaptation phase
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                log_prob_array,
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_9(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
        """Run MCMC sampling using BlackJAX."""
        rng_key = random.PRNGKey(self.random_seed)

        # Convert parameter dict to array for sampling
        param_names = list(initial_params.keys())
        initial_position = jnp.array([initial_params[name] for name in param_names])

        # Create log prob function that takes array input
        def log_prob_array(position_array):
            params_dict = {name: position_array[i] for i, name in enumerate(param_names)}
            return log_prob_fn(None)

        try:
            # Adaptation phase
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                log_prob_array,
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_10(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
            warmup = None

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_11(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
                None,
                log_prob_array,
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_12(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
                None,
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_13(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
                target_acceptance_rate=None,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_14(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
                log_prob_array,
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_15(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
                target_acceptance_rate=self.target_accept_rate,
            )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_16(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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
                )

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_17(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = None

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_18(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(None, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_19(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, None, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_20(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=None)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_21(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_22(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_23(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, )

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_24(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = None

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_25(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(None, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_26(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(**parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_27(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, )

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_28(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = None
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_29(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = None
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_30(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(None, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_31(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, None)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_32(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_33(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, )
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_34(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = None
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_35(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(None):
                chain_key = random.fold_in(rng_key, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_36(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = None
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_37(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(None, chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_38(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, None)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_39(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(chain_id)
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_40(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

            # Sampling phase
            sampler = blackjax.nuts(log_prob_array, **parameters)

            def one_step(carry, rng_key):
                state, _ = carry
                new_state, info = sampler.step(rng_key, state)
                return (new_state, info), new_state.position

            # Run chains
            all_samples = []
            for chain_id in range(self.num_chains):
                chain_key = random.fold_in(rng_key, )
                sample_keys = random.split(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_41(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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
                sample_keys = None

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_42(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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
                sample_keys = random.split(None, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_43(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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
                sample_keys = random.split(chain_key, None)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_44(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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
                sample_keys = random.split(self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_45(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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
                sample_keys = random.split(chain_key, )

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_46(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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
                sample_keys = random.rsplit(chain_key, self.num_samples)

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_47(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = None
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_48(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(None, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_49(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, None, sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_50(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), None)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_51(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan((state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_52(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_53(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), )
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_54(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(None)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_55(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = None  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_56(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(None)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_57(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = None

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_58(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(None)}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_59(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = None

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_60(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "XXsamplesXX": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_61(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "SAMPLES": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_62(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "XXparam_namesXX": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_63(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "PARAM_NAMES": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_64(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "XXfinal_stateXX": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_65(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "FINAL_STATE": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_66(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(None, UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_67(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", None)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_68(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_69(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", )
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_70(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = None

    def xǁBayesianFitterǁ_run_mcmc__mutmut_71(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full(None, value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_72(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), None) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_73(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full(value) for name, value in initial_params.items()
            }

    def xǁBayesianFitterǁ_run_mcmc__mutmut_74(self, log_prob_fn: Callable, initial_params: dict[str, float]) -> None:
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

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=self.num_warmup)

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

                (final_state, _), chain_samples = jax.lax.scan(one_step, (state, None), sample_keys)
                all_samples.append(chain_samples)

            # Store results
            samples_array = jnp.stack(all_samples)  # Shape: (num_chains, num_samples, num_params)

            # Convert back to parameter dictionaries
            self.posterior_samples_ = {param_names[i]: samples_array[:, :, i] for i in range(len(param_names))}

            self.mcmc_results_ = {
                "samples": samples_array,
                "param_names": param_names,
                "final_state": final_state,
            }

        except Exception as e:
            warnings.warn(f"MCMC sampling failed: {e!s}. Using point estimates.", UserWarning)
            # Fallback to point estimates
            self.posterior_samples_ = {
                name: jnp.full((self.num_chains, self.num_samples), ) for name, value in initial_params.items()
            }
    
    xǁBayesianFitterǁ_run_mcmc__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ_run_mcmc__mutmut_1': xǁBayesianFitterǁ_run_mcmc__mutmut_1, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_2': xǁBayesianFitterǁ_run_mcmc__mutmut_2, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_3': xǁBayesianFitterǁ_run_mcmc__mutmut_3, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_4': xǁBayesianFitterǁ_run_mcmc__mutmut_4, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_5': xǁBayesianFitterǁ_run_mcmc__mutmut_5, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_6': xǁBayesianFitterǁ_run_mcmc__mutmut_6, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_7': xǁBayesianFitterǁ_run_mcmc__mutmut_7, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_8': xǁBayesianFitterǁ_run_mcmc__mutmut_8, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_9': xǁBayesianFitterǁ_run_mcmc__mutmut_9, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_10': xǁBayesianFitterǁ_run_mcmc__mutmut_10, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_11': xǁBayesianFitterǁ_run_mcmc__mutmut_11, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_12': xǁBayesianFitterǁ_run_mcmc__mutmut_12, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_13': xǁBayesianFitterǁ_run_mcmc__mutmut_13, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_14': xǁBayesianFitterǁ_run_mcmc__mutmut_14, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_15': xǁBayesianFitterǁ_run_mcmc__mutmut_15, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_16': xǁBayesianFitterǁ_run_mcmc__mutmut_16, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_17': xǁBayesianFitterǁ_run_mcmc__mutmut_17, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_18': xǁBayesianFitterǁ_run_mcmc__mutmut_18, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_19': xǁBayesianFitterǁ_run_mcmc__mutmut_19, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_20': xǁBayesianFitterǁ_run_mcmc__mutmut_20, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_21': xǁBayesianFitterǁ_run_mcmc__mutmut_21, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_22': xǁBayesianFitterǁ_run_mcmc__mutmut_22, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_23': xǁBayesianFitterǁ_run_mcmc__mutmut_23, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_24': xǁBayesianFitterǁ_run_mcmc__mutmut_24, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_25': xǁBayesianFitterǁ_run_mcmc__mutmut_25, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_26': xǁBayesianFitterǁ_run_mcmc__mutmut_26, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_27': xǁBayesianFitterǁ_run_mcmc__mutmut_27, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_28': xǁBayesianFitterǁ_run_mcmc__mutmut_28, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_29': xǁBayesianFitterǁ_run_mcmc__mutmut_29, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_30': xǁBayesianFitterǁ_run_mcmc__mutmut_30, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_31': xǁBayesianFitterǁ_run_mcmc__mutmut_31, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_32': xǁBayesianFitterǁ_run_mcmc__mutmut_32, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_33': xǁBayesianFitterǁ_run_mcmc__mutmut_33, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_34': xǁBayesianFitterǁ_run_mcmc__mutmut_34, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_35': xǁBayesianFitterǁ_run_mcmc__mutmut_35, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_36': xǁBayesianFitterǁ_run_mcmc__mutmut_36, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_37': xǁBayesianFitterǁ_run_mcmc__mutmut_37, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_38': xǁBayesianFitterǁ_run_mcmc__mutmut_38, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_39': xǁBayesianFitterǁ_run_mcmc__mutmut_39, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_40': xǁBayesianFitterǁ_run_mcmc__mutmut_40, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_41': xǁBayesianFitterǁ_run_mcmc__mutmut_41, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_42': xǁBayesianFitterǁ_run_mcmc__mutmut_42, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_43': xǁBayesianFitterǁ_run_mcmc__mutmut_43, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_44': xǁBayesianFitterǁ_run_mcmc__mutmut_44, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_45': xǁBayesianFitterǁ_run_mcmc__mutmut_45, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_46': xǁBayesianFitterǁ_run_mcmc__mutmut_46, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_47': xǁBayesianFitterǁ_run_mcmc__mutmut_47, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_48': xǁBayesianFitterǁ_run_mcmc__mutmut_48, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_49': xǁBayesianFitterǁ_run_mcmc__mutmut_49, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_50': xǁBayesianFitterǁ_run_mcmc__mutmut_50, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_51': xǁBayesianFitterǁ_run_mcmc__mutmut_51, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_52': xǁBayesianFitterǁ_run_mcmc__mutmut_52, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_53': xǁBayesianFitterǁ_run_mcmc__mutmut_53, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_54': xǁBayesianFitterǁ_run_mcmc__mutmut_54, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_55': xǁBayesianFitterǁ_run_mcmc__mutmut_55, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_56': xǁBayesianFitterǁ_run_mcmc__mutmut_56, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_57': xǁBayesianFitterǁ_run_mcmc__mutmut_57, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_58': xǁBayesianFitterǁ_run_mcmc__mutmut_58, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_59': xǁBayesianFitterǁ_run_mcmc__mutmut_59, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_60': xǁBayesianFitterǁ_run_mcmc__mutmut_60, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_61': xǁBayesianFitterǁ_run_mcmc__mutmut_61, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_62': xǁBayesianFitterǁ_run_mcmc__mutmut_62, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_63': xǁBayesianFitterǁ_run_mcmc__mutmut_63, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_64': xǁBayesianFitterǁ_run_mcmc__mutmut_64, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_65': xǁBayesianFitterǁ_run_mcmc__mutmut_65, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_66': xǁBayesianFitterǁ_run_mcmc__mutmut_66, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_67': xǁBayesianFitterǁ_run_mcmc__mutmut_67, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_68': xǁBayesianFitterǁ_run_mcmc__mutmut_68, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_69': xǁBayesianFitterǁ_run_mcmc__mutmut_69, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_70': xǁBayesianFitterǁ_run_mcmc__mutmut_70, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_71': xǁBayesianFitterǁ_run_mcmc__mutmut_71, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_72': xǁBayesianFitterǁ_run_mcmc__mutmut_72, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_73': xǁBayesianFitterǁ_run_mcmc__mutmut_73, 
        'xǁBayesianFitterǁ_run_mcmc__mutmut_74': xǁBayesianFitterǁ_run_mcmc__mutmut_74
    }
    xǁBayesianFitterǁ_run_mcmc__mutmut_orig.__name__ = 'xǁBayesianFitterǁ_run_mcmc'

    def get_parameter_estimates(self) -> dict[str, float]:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁget_parameter_estimates__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁget_parameter_estimates__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_orig(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        return {param: float(jnp.mean(samples)) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_1(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is not None:
            raise RuntimeError("Model has not been fitted yet.")

        return {param: float(jnp.mean(samples)) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_2(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError(None)

        return {param: float(jnp.mean(samples)) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_3(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("XXModel has not been fitted yet.XX")

        return {param: float(jnp.mean(samples)) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_4(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("model has not been fitted yet.")

        return {param: float(jnp.mean(samples)) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_5(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")

        return {param: float(jnp.mean(samples)) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_6(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        return {param: float(None) for param, samples in self.posterior_samples_.items()}

    def xǁBayesianFitterǁget_parameter_estimates__mutmut_7(self) -> dict[str, float]:
        """Get posterior mean estimates for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        return {param: float(jnp.mean(None)) for param, samples in self.posterior_samples_.items()}
    
    xǁBayesianFitterǁget_parameter_estimates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁget_parameter_estimates__mutmut_1': xǁBayesianFitterǁget_parameter_estimates__mutmut_1, 
        'xǁBayesianFitterǁget_parameter_estimates__mutmut_2': xǁBayesianFitterǁget_parameter_estimates__mutmut_2, 
        'xǁBayesianFitterǁget_parameter_estimates__mutmut_3': xǁBayesianFitterǁget_parameter_estimates__mutmut_3, 
        'xǁBayesianFitterǁget_parameter_estimates__mutmut_4': xǁBayesianFitterǁget_parameter_estimates__mutmut_4, 
        'xǁBayesianFitterǁget_parameter_estimates__mutmut_5': xǁBayesianFitterǁget_parameter_estimates__mutmut_5, 
        'xǁBayesianFitterǁget_parameter_estimates__mutmut_6': xǁBayesianFitterǁget_parameter_estimates__mutmut_6, 
        'xǁBayesianFitterǁget_parameter_estimates__mutmut_7': xǁBayesianFitterǁget_parameter_estimates__mutmut_7
    }
    xǁBayesianFitterǁget_parameter_estimates__mutmut_orig.__name__ = 'xǁBayesianFitterǁget_parameter_estimates'

    def get_confidence_intervals(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        args = [credible_mass]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁget_confidence_intervals__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁget_confidence_intervals__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_orig(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_1(self, credible_mass: float = 1.95) -> dict[str, tuple[float, float]]:
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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_2(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is not None:
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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_3(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError(None)

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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_4(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("XXModel has not been fitted yet.XX")

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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_5(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("model has not been fitted yet.")

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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_6(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")

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

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_7(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = None
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_8(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 + credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_9(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 2 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_10(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = None
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_11(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha * 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_12(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 / alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_13(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 101 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_14(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 3
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_15(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = None

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_16(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 / (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_17(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 101 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_18(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 + alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_19(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (2 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_20(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha * 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_21(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 3)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_22(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = None
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_23(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = None
            lower = float(jnp.percentile(flat_samples, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_24(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = None
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_25(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(None)
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_26(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(None, lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_27(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, None))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_28(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(lower_percentile))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_29(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
        """Get credible intervals for parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        alpha = 1 - credible_mass
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        intervals = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            lower = float(jnp.percentile(flat_samples, ))
            upper = float(jnp.percentile(flat_samples, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_30(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            upper = None
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_31(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            upper = float(None)
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_32(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            upper = float(jnp.percentile(None, upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_33(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            upper = float(jnp.percentile(flat_samples, None))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_34(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            upper = float(jnp.percentile(upper_percentile))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_35(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            upper = float(jnp.percentile(flat_samples, ))
            intervals[param] = (lower, upper)

        return intervals

    def xǁBayesianFitterǁget_confidence_intervals__mutmut_36(self, credible_mass: float = 0.95) -> dict[str, tuple[float, float]]:
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
            intervals[param] = None

        return intervals
    
    xǁBayesianFitterǁget_confidence_intervals__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁget_confidence_intervals__mutmut_1': xǁBayesianFitterǁget_confidence_intervals__mutmut_1, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_2': xǁBayesianFitterǁget_confidence_intervals__mutmut_2, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_3': xǁBayesianFitterǁget_confidence_intervals__mutmut_3, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_4': xǁBayesianFitterǁget_confidence_intervals__mutmut_4, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_5': xǁBayesianFitterǁget_confidence_intervals__mutmut_5, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_6': xǁBayesianFitterǁget_confidence_intervals__mutmut_6, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_7': xǁBayesianFitterǁget_confidence_intervals__mutmut_7, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_8': xǁBayesianFitterǁget_confidence_intervals__mutmut_8, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_9': xǁBayesianFitterǁget_confidence_intervals__mutmut_9, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_10': xǁBayesianFitterǁget_confidence_intervals__mutmut_10, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_11': xǁBayesianFitterǁget_confidence_intervals__mutmut_11, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_12': xǁBayesianFitterǁget_confidence_intervals__mutmut_12, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_13': xǁBayesianFitterǁget_confidence_intervals__mutmut_13, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_14': xǁBayesianFitterǁget_confidence_intervals__mutmut_14, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_15': xǁBayesianFitterǁget_confidence_intervals__mutmut_15, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_16': xǁBayesianFitterǁget_confidence_intervals__mutmut_16, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_17': xǁBayesianFitterǁget_confidence_intervals__mutmut_17, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_18': xǁBayesianFitterǁget_confidence_intervals__mutmut_18, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_19': xǁBayesianFitterǁget_confidence_intervals__mutmut_19, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_20': xǁBayesianFitterǁget_confidence_intervals__mutmut_20, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_21': xǁBayesianFitterǁget_confidence_intervals__mutmut_21, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_22': xǁBayesianFitterǁget_confidence_intervals__mutmut_22, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_23': xǁBayesianFitterǁget_confidence_intervals__mutmut_23, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_24': xǁBayesianFitterǁget_confidence_intervals__mutmut_24, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_25': xǁBayesianFitterǁget_confidence_intervals__mutmut_25, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_26': xǁBayesianFitterǁget_confidence_intervals__mutmut_26, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_27': xǁBayesianFitterǁget_confidence_intervals__mutmut_27, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_28': xǁBayesianFitterǁget_confidence_intervals__mutmut_28, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_29': xǁBayesianFitterǁget_confidence_intervals__mutmut_29, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_30': xǁBayesianFitterǁget_confidence_intervals__mutmut_30, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_31': xǁBayesianFitterǁget_confidence_intervals__mutmut_31, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_32': xǁBayesianFitterǁget_confidence_intervals__mutmut_32, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_33': xǁBayesianFitterǁget_confidence_intervals__mutmut_33, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_34': xǁBayesianFitterǁget_confidence_intervals__mutmut_34, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_35': xǁBayesianFitterǁget_confidence_intervals__mutmut_35, 
        'xǁBayesianFitterǁget_confidence_intervals__mutmut_36': xǁBayesianFitterǁget_confidence_intervals__mutmut_36
    }
    xǁBayesianFitterǁget_confidence_intervals__mutmut_orig.__name__ = 'xǁBayesianFitterǁget_confidence_intervals'

    def get_summary(self) -> dict[str, dict[str, float]]:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁget_summary__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁget_summary__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁget_summary__mutmut_orig(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_1(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is not None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_2(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError(None)

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_3(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("XXModel has not been fitted yet.XX")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_4(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_5(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_6(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = None
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_7(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = None
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_8(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = None

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_9(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "XXmeanXX": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_10(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "MEAN": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_11(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(None),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_12(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(None)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_13(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "XXstdXX": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_14(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "STD": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_15(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(None),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_16(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(None)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_17(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "XXmedianXX": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_18(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "MEDIAN": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_19(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(None),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_20(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(None)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_21(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "XX2.5%XX": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_22(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(None),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_23(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(None, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_24(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, None)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_25(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_26(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, )),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_27(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 3.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_28(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "XX97.5%XX": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_29(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(None),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_30(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(None, 97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_31(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, None)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_32(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(97.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_33(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, )),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_34(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 98.5)),
                "n_eff": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_35(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "XXn_effXX": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_36(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "N_EFF": float(len(flat_samples)),  # Simplified
            }

        return summary

    def xǁBayesianFitterǁget_summary__mutmut_37(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all parameters."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            flat_samples = samples.flatten()
            summary[param] = {
                "mean": float(jnp.mean(flat_samples)),
                "std": float(jnp.std(flat_samples)),
                "median": float(jnp.median(flat_samples)),
                "2.5%": float(jnp.percentile(flat_samples, 2.5)),
                "97.5%": float(jnp.percentile(flat_samples, 97.5)),
                "n_eff": float(None),  # Simplified
            }

        return summary
    
    xǁBayesianFitterǁget_summary__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁget_summary__mutmut_1': xǁBayesianFitterǁget_summary__mutmut_1, 
        'xǁBayesianFitterǁget_summary__mutmut_2': xǁBayesianFitterǁget_summary__mutmut_2, 
        'xǁBayesianFitterǁget_summary__mutmut_3': xǁBayesianFitterǁget_summary__mutmut_3, 
        'xǁBayesianFitterǁget_summary__mutmut_4': xǁBayesianFitterǁget_summary__mutmut_4, 
        'xǁBayesianFitterǁget_summary__mutmut_5': xǁBayesianFitterǁget_summary__mutmut_5, 
        'xǁBayesianFitterǁget_summary__mutmut_6': xǁBayesianFitterǁget_summary__mutmut_6, 
        'xǁBayesianFitterǁget_summary__mutmut_7': xǁBayesianFitterǁget_summary__mutmut_7, 
        'xǁBayesianFitterǁget_summary__mutmut_8': xǁBayesianFitterǁget_summary__mutmut_8, 
        'xǁBayesianFitterǁget_summary__mutmut_9': xǁBayesianFitterǁget_summary__mutmut_9, 
        'xǁBayesianFitterǁget_summary__mutmut_10': xǁBayesianFitterǁget_summary__mutmut_10, 
        'xǁBayesianFitterǁget_summary__mutmut_11': xǁBayesianFitterǁget_summary__mutmut_11, 
        'xǁBayesianFitterǁget_summary__mutmut_12': xǁBayesianFitterǁget_summary__mutmut_12, 
        'xǁBayesianFitterǁget_summary__mutmut_13': xǁBayesianFitterǁget_summary__mutmut_13, 
        'xǁBayesianFitterǁget_summary__mutmut_14': xǁBayesianFitterǁget_summary__mutmut_14, 
        'xǁBayesianFitterǁget_summary__mutmut_15': xǁBayesianFitterǁget_summary__mutmut_15, 
        'xǁBayesianFitterǁget_summary__mutmut_16': xǁBayesianFitterǁget_summary__mutmut_16, 
        'xǁBayesianFitterǁget_summary__mutmut_17': xǁBayesianFitterǁget_summary__mutmut_17, 
        'xǁBayesianFitterǁget_summary__mutmut_18': xǁBayesianFitterǁget_summary__mutmut_18, 
        'xǁBayesianFitterǁget_summary__mutmut_19': xǁBayesianFitterǁget_summary__mutmut_19, 
        'xǁBayesianFitterǁget_summary__mutmut_20': xǁBayesianFitterǁget_summary__mutmut_20, 
        'xǁBayesianFitterǁget_summary__mutmut_21': xǁBayesianFitterǁget_summary__mutmut_21, 
        'xǁBayesianFitterǁget_summary__mutmut_22': xǁBayesianFitterǁget_summary__mutmut_22, 
        'xǁBayesianFitterǁget_summary__mutmut_23': xǁBayesianFitterǁget_summary__mutmut_23, 
        'xǁBayesianFitterǁget_summary__mutmut_24': xǁBayesianFitterǁget_summary__mutmut_24, 
        'xǁBayesianFitterǁget_summary__mutmut_25': xǁBayesianFitterǁget_summary__mutmut_25, 
        'xǁBayesianFitterǁget_summary__mutmut_26': xǁBayesianFitterǁget_summary__mutmut_26, 
        'xǁBayesianFitterǁget_summary__mutmut_27': xǁBayesianFitterǁget_summary__mutmut_27, 
        'xǁBayesianFitterǁget_summary__mutmut_28': xǁBayesianFitterǁget_summary__mutmut_28, 
        'xǁBayesianFitterǁget_summary__mutmut_29': xǁBayesianFitterǁget_summary__mutmut_29, 
        'xǁBayesianFitterǁget_summary__mutmut_30': xǁBayesianFitterǁget_summary__mutmut_30, 
        'xǁBayesianFitterǁget_summary__mutmut_31': xǁBayesianFitterǁget_summary__mutmut_31, 
        'xǁBayesianFitterǁget_summary__mutmut_32': xǁBayesianFitterǁget_summary__mutmut_32, 
        'xǁBayesianFitterǁget_summary__mutmut_33': xǁBayesianFitterǁget_summary__mutmut_33, 
        'xǁBayesianFitterǁget_summary__mutmut_34': xǁBayesianFitterǁget_summary__mutmut_34, 
        'xǁBayesianFitterǁget_summary__mutmut_35': xǁBayesianFitterǁget_summary__mutmut_35, 
        'xǁBayesianFitterǁget_summary__mutmut_36': xǁBayesianFitterǁget_summary__mutmut_36, 
        'xǁBayesianFitterǁget_summary__mutmut_37': xǁBayesianFitterǁget_summary__mutmut_37
    }
    xǁBayesianFitterǁget_summary__mutmut_orig.__name__ = 'xǁBayesianFitterǁget_summary'

    def plot_trace(self, **kwargs):
        args = []# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁplot_trace__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁplot_trace__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁplot_trace__mutmut_orig(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_1(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is not None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_2(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError(None)

        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_3(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("XXModel has not been fitted yet.XX")

        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_4(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_5(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")

        idata = self._to_inference_data()
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_6(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = None
        return az.plot_trace(idata, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_7(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_trace(None, **kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_8(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_trace(**kwargs)

    def xǁBayesianFitterǁplot_trace__mutmut_9(self, **kwargs):
        """Plot MCMC traces using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_trace(idata, )
    
    xǁBayesianFitterǁplot_trace__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁplot_trace__mutmut_1': xǁBayesianFitterǁplot_trace__mutmut_1, 
        'xǁBayesianFitterǁplot_trace__mutmut_2': xǁBayesianFitterǁplot_trace__mutmut_2, 
        'xǁBayesianFitterǁplot_trace__mutmut_3': xǁBayesianFitterǁplot_trace__mutmut_3, 
        'xǁBayesianFitterǁplot_trace__mutmut_4': xǁBayesianFitterǁplot_trace__mutmut_4, 
        'xǁBayesianFitterǁplot_trace__mutmut_5': xǁBayesianFitterǁplot_trace__mutmut_5, 
        'xǁBayesianFitterǁplot_trace__mutmut_6': xǁBayesianFitterǁplot_trace__mutmut_6, 
        'xǁBayesianFitterǁplot_trace__mutmut_7': xǁBayesianFitterǁplot_trace__mutmut_7, 
        'xǁBayesianFitterǁplot_trace__mutmut_8': xǁBayesianFitterǁplot_trace__mutmut_8, 
        'xǁBayesianFitterǁplot_trace__mutmut_9': xǁBayesianFitterǁplot_trace__mutmut_9
    }
    xǁBayesianFitterǁplot_trace__mutmut_orig.__name__ = 'xǁBayesianFitterǁplot_trace'

    def plot_posterior(self, **kwargs):
        args = []# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁplot_posterior__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁplot_posterior__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁplot_posterior__mutmut_orig(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_1(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is not None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_2(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError(None)

        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_3(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("XXModel has not been fitted yet.XX")

        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_4(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_5(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET.")

        idata = self._to_inference_data()
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_6(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = None
        return az.plot_posterior(idata, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_7(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_posterior(None, **kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_8(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_posterior(**kwargs)

    def xǁBayesianFitterǁplot_posterior__mutmut_9(self, **kwargs):
        """Plot posterior distributions using arviz."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        idata = self._to_inference_data()
        return az.plot_posterior(idata, )
    
    xǁBayesianFitterǁplot_posterior__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁplot_posterior__mutmut_1': xǁBayesianFitterǁplot_posterior__mutmut_1, 
        'xǁBayesianFitterǁplot_posterior__mutmut_2': xǁBayesianFitterǁplot_posterior__mutmut_2, 
        'xǁBayesianFitterǁplot_posterior__mutmut_3': xǁBayesianFitterǁplot_posterior__mutmut_3, 
        'xǁBayesianFitterǁplot_posterior__mutmut_4': xǁBayesianFitterǁplot_posterior__mutmut_4, 
        'xǁBayesianFitterǁplot_posterior__mutmut_5': xǁBayesianFitterǁplot_posterior__mutmut_5, 
        'xǁBayesianFitterǁplot_posterior__mutmut_6': xǁBayesianFitterǁplot_posterior__mutmut_6, 
        'xǁBayesianFitterǁplot_posterior__mutmut_7': xǁBayesianFitterǁplot_posterior__mutmut_7, 
        'xǁBayesianFitterǁplot_posterior__mutmut_8': xǁBayesianFitterǁplot_posterior__mutmut_8, 
        'xǁBayesianFitterǁplot_posterior__mutmut_9': xǁBayesianFitterǁplot_posterior__mutmut_9
    }
    xǁBayesianFitterǁplot_posterior__mutmut_orig.__name__ = 'xǁBayesianFitterǁplot_posterior'

    def _to_inference_data(self) -> az.InferenceData:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBayesianFitterǁ_to_inference_data__mutmut_orig'), object.__getattribute__(self, 'xǁBayesianFitterǁ_to_inference_data__mutmut_mutants'), args, kwargs, self)

    def xǁBayesianFitterǁ_to_inference_data__mutmut_orig(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"chain": range(self.num_chains), "draw": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_1(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=None, coords={"chain": range(self.num_chains), "draw": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_2(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords=None
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_3(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            coords={"chain": range(self.num_chains), "draw": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_4(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_5(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"XXchainXX": range(self.num_chains), "draw": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_6(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"CHAIN": range(self.num_chains), "draw": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_7(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"chain": range(None), "draw": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_8(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"chain": range(self.num_chains), "XXdrawXX": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_9(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"chain": range(self.num_chains), "DRAW": range(self.num_samples)}
        )

    def xǁBayesianFitterǁ_to_inference_data__mutmut_10(self) -> az.InferenceData:
        """Convert samples to arviz InferenceData format."""
        return az.from_dict(
            posterior=self.posterior_samples_, coords={"chain": range(self.num_chains), "draw": range(None)}
        )
    
    xǁBayesianFitterǁ_to_inference_data__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBayesianFitterǁ_to_inference_data__mutmut_1': xǁBayesianFitterǁ_to_inference_data__mutmut_1, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_2': xǁBayesianFitterǁ_to_inference_data__mutmut_2, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_3': xǁBayesianFitterǁ_to_inference_data__mutmut_3, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_4': xǁBayesianFitterǁ_to_inference_data__mutmut_4, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_5': xǁBayesianFitterǁ_to_inference_data__mutmut_5, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_6': xǁBayesianFitterǁ_to_inference_data__mutmut_6, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_7': xǁBayesianFitterǁ_to_inference_data__mutmut_7, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_8': xǁBayesianFitterǁ_to_inference_data__mutmut_8, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_9': xǁBayesianFitterǁ_to_inference_data__mutmut_9, 
        'xǁBayesianFitterǁ_to_inference_data__mutmut_10': xǁBayesianFitterǁ_to_inference_data__mutmut_10
    }
    xǁBayesianFitterǁ_to_inference_data__mutmut_orig.__name__ = 'xǁBayesianFitterǁ_to_inference_data'
