"""A Bayesian fitter that uses the BlackJAX library for MCMC sampling."""

from collections.abc import Callable
from typing import Any

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
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
        args = [model, num_chains, num_warmup, num_samples]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁ__init____mutmut_orig(
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

    def xǁBlackJaxFitterǁ__init____mutmut_1(
        self,
        model: Any,
        num_chains: int = 5,
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

    def xǁBlackJaxFitterǁ__init____mutmut_2(
        self,
        model: Any,
        num_chains: int = 4,
        num_warmup: int = 1001,
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

    def xǁBlackJaxFitterǁ__init____mutmut_3(
        self,
        model: Any,
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1001,
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

    def xǁBlackJaxFitterǁ__init____mutmut_4(
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
        self.model = None
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.states = None
        self.kernel = blackjax.nuts

    def xǁBlackJaxFitterǁ__init____mutmut_5(
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
        self.num_chains = None
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.states = None
        self.kernel = blackjax.nuts

    def xǁBlackJaxFitterǁ__init____mutmut_6(
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
        self.num_warmup = None
        self.num_samples = num_samples
        self.states = None
        self.kernel = blackjax.nuts

    def xǁBlackJaxFitterǁ__init____mutmut_7(
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
        self.num_samples = None
        self.states = None
        self.kernel = blackjax.nuts

    def xǁBlackJaxFitterǁ__init____mutmut_8(
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
        self.states = ""
        self.kernel = blackjax.nuts

    def xǁBlackJaxFitterǁ__init____mutmut_9(
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
        self.kernel = None
    
    xǁBlackJaxFitterǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁ__init____mutmut_1': xǁBlackJaxFitterǁ__init____mutmut_1, 
        'xǁBlackJaxFitterǁ__init____mutmut_2': xǁBlackJaxFitterǁ__init____mutmut_2, 
        'xǁBlackJaxFitterǁ__init____mutmut_3': xǁBlackJaxFitterǁ__init____mutmut_3, 
        'xǁBlackJaxFitterǁ__init____mutmut_4': xǁBlackJaxFitterǁ__init____mutmut_4, 
        'xǁBlackJaxFitterǁ__init____mutmut_5': xǁBlackJaxFitterǁ__init____mutmut_5, 
        'xǁBlackJaxFitterǁ__init____mutmut_6': xǁBlackJaxFitterǁ__init____mutmut_6, 
        'xǁBlackJaxFitterǁ__init____mutmut_7': xǁBlackJaxFitterǁ__init____mutmut_7, 
        'xǁBlackJaxFitterǁ__init____mutmut_8': xǁBlackJaxFitterǁ__init____mutmut_8, 
        'xǁBlackJaxFitterǁ__init____mutmut_9': xǁBlackJaxFitterǁ__init____mutmut_9
    }
    xǁBlackJaxFitterǁ__init____mutmut_orig.__name__ = 'xǁBlackJaxFitterǁ__init__'

    def fit(
        self,
        y: jnp.ndarray,
        log_probability_fn: Callable,
        initial_params: dict[str, float],
        **kwargs: Any,
    ) -> None:
        args = [y, log_probability_fn, initial_params]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁfit__mutmut_orig(
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

    def xǁBlackJaxFitterǁfit__mutmut_1(
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
        rng_key = None

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

    def xǁBlackJaxFitterǁfit__mutmut_2(
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
        rng_key = jax.random.PRNGKey(None)

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

    def xǁBlackJaxFitterǁfit__mutmut_3(
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
        rng_key = jax.random.PRNGKey(1)

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

    def xǁBlackJaxFitterǁfit__mutmut_4(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = None
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

    def xǁBlackJaxFitterǁfit__mutmut_5(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(None, num_samples)
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

    def xǁBlackJaxFitterǁfit__mutmut_6(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, None)
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

    def xǁBlackJaxFitterǁfit__mutmut_7(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(num_samples)
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

    def xǁBlackJaxFitterǁfit__mutmut_8(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, )
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

    def xǁBlackJaxFitterǁfit__mutmut_9(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.rsplit(rng_key, num_samples)
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

    def xǁBlackJaxFitterǁfit__mutmut_10(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = None
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

    def xǁBlackJaxFitterǁfit__mutmut_11(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(None, initial_state, keys)
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

    def xǁBlackJaxFitterǁfit__mutmut_12(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, None, keys)
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

    def xǁBlackJaxFitterǁfit__mutmut_13(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, None)
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

    def xǁBlackJaxFitterǁfit__mutmut_14(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(initial_state, keys)
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

    def xǁBlackJaxFitterǁfit__mutmut_15(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, keys)
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

    def xǁBlackJaxFitterǁfit__mutmut_16(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, )
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

    def xǁBlackJaxFitterǁfit__mutmut_17(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)
            return states

        all_states = None
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

    def xǁBlackJaxFitterǁfit__mutmut_18(
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)
            return states

        all_states = []
        for i in range(None):
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

    def xǁBlackJaxFitterǁfit__mutmut_19(
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
            chain_rng_key, rng_key = None

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

    def xǁBlackJaxFitterǁfit__mutmut_20(
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
            chain_rng_key, rng_key = jax.random.split(None)

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

    def xǁBlackJaxFitterǁfit__mutmut_21(
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
            adapt = None
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

    def xǁBlackJaxFitterǁfit__mutmut_22(
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
            adapt = blackjax.window_adaptation(None, log_probability_fn)
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

    def xǁBlackJaxFitterǁfit__mutmut_23(
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
            adapt = blackjax.window_adaptation(blackjax.nuts, None)
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

    def xǁBlackJaxFitterǁfit__mutmut_24(
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
            adapt = blackjax.window_adaptation(log_probability_fn)
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

    def xǁBlackJaxFitterǁfit__mutmut_25(
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
            adapt = blackjax.window_adaptation(blackjax.nuts, )
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

    def xǁBlackJaxFitterǁfit__mutmut_26(
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
            (last_state, parameters), _ = None

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_27(
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
                None,
                initial_params,
                self.num_warmup,
            )

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_28(
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
                None,
                self.num_warmup,
            )

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_29(
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
                None,
            )

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_30(
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
                initial_params,
                self.num_warmup,
            )

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_31(
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
                self.num_warmup,
            )

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_32(
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
                )

            # Sample from the posterior
            kernel = blackjax.nuts(log_probability_fn, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_33(
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
            kernel = None
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_34(
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
            kernel = blackjax.nuts(None, **parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_35(
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
            kernel = blackjax.nuts(**parameters)
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_36(
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
            kernel = blackjax.nuts(log_probability_fn, )
            states = inference_loop(chain_rng_key, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_37(
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
            states = None
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_38(
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
            states = inference_loop(None, kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_39(
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
            states = inference_loop(chain_rng_key, None, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_40(
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
            states = inference_loop(chain_rng_key, kernel, None, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_41(
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
            states = inference_loop(chain_rng_key, kernel, last_state, None)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_42(
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
            states = inference_loop(kernel, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_43(
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
            states = inference_loop(chain_rng_key, last_state, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_44(
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
            states = inference_loop(chain_rng_key, kernel, self.num_samples)
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_45(
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
            states = inference_loop(chain_rng_key, kernel, last_state, )
            all_states.append(states)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_46(
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
            all_states.append(None)

        self.states = all_states

    def xǁBlackJaxFitterǁfit__mutmut_47(
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

        self.states = None
    
    xǁBlackJaxFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁfit__mutmut_1': xǁBlackJaxFitterǁfit__mutmut_1, 
        'xǁBlackJaxFitterǁfit__mutmut_2': xǁBlackJaxFitterǁfit__mutmut_2, 
        'xǁBlackJaxFitterǁfit__mutmut_3': xǁBlackJaxFitterǁfit__mutmut_3, 
        'xǁBlackJaxFitterǁfit__mutmut_4': xǁBlackJaxFitterǁfit__mutmut_4, 
        'xǁBlackJaxFitterǁfit__mutmut_5': xǁBlackJaxFitterǁfit__mutmut_5, 
        'xǁBlackJaxFitterǁfit__mutmut_6': xǁBlackJaxFitterǁfit__mutmut_6, 
        'xǁBlackJaxFitterǁfit__mutmut_7': xǁBlackJaxFitterǁfit__mutmut_7, 
        'xǁBlackJaxFitterǁfit__mutmut_8': xǁBlackJaxFitterǁfit__mutmut_8, 
        'xǁBlackJaxFitterǁfit__mutmut_9': xǁBlackJaxFitterǁfit__mutmut_9, 
        'xǁBlackJaxFitterǁfit__mutmut_10': xǁBlackJaxFitterǁfit__mutmut_10, 
        'xǁBlackJaxFitterǁfit__mutmut_11': xǁBlackJaxFitterǁfit__mutmut_11, 
        'xǁBlackJaxFitterǁfit__mutmut_12': xǁBlackJaxFitterǁfit__mutmut_12, 
        'xǁBlackJaxFitterǁfit__mutmut_13': xǁBlackJaxFitterǁfit__mutmut_13, 
        'xǁBlackJaxFitterǁfit__mutmut_14': xǁBlackJaxFitterǁfit__mutmut_14, 
        'xǁBlackJaxFitterǁfit__mutmut_15': xǁBlackJaxFitterǁfit__mutmut_15, 
        'xǁBlackJaxFitterǁfit__mutmut_16': xǁBlackJaxFitterǁfit__mutmut_16, 
        'xǁBlackJaxFitterǁfit__mutmut_17': xǁBlackJaxFitterǁfit__mutmut_17, 
        'xǁBlackJaxFitterǁfit__mutmut_18': xǁBlackJaxFitterǁfit__mutmut_18, 
        'xǁBlackJaxFitterǁfit__mutmut_19': xǁBlackJaxFitterǁfit__mutmut_19, 
        'xǁBlackJaxFitterǁfit__mutmut_20': xǁBlackJaxFitterǁfit__mutmut_20, 
        'xǁBlackJaxFitterǁfit__mutmut_21': xǁBlackJaxFitterǁfit__mutmut_21, 
        'xǁBlackJaxFitterǁfit__mutmut_22': xǁBlackJaxFitterǁfit__mutmut_22, 
        'xǁBlackJaxFitterǁfit__mutmut_23': xǁBlackJaxFitterǁfit__mutmut_23, 
        'xǁBlackJaxFitterǁfit__mutmut_24': xǁBlackJaxFitterǁfit__mutmut_24, 
        'xǁBlackJaxFitterǁfit__mutmut_25': xǁBlackJaxFitterǁfit__mutmut_25, 
        'xǁBlackJaxFitterǁfit__mutmut_26': xǁBlackJaxFitterǁfit__mutmut_26, 
        'xǁBlackJaxFitterǁfit__mutmut_27': xǁBlackJaxFitterǁfit__mutmut_27, 
        'xǁBlackJaxFitterǁfit__mutmut_28': xǁBlackJaxFitterǁfit__mutmut_28, 
        'xǁBlackJaxFitterǁfit__mutmut_29': xǁBlackJaxFitterǁfit__mutmut_29, 
        'xǁBlackJaxFitterǁfit__mutmut_30': xǁBlackJaxFitterǁfit__mutmut_30, 
        'xǁBlackJaxFitterǁfit__mutmut_31': xǁBlackJaxFitterǁfit__mutmut_31, 
        'xǁBlackJaxFitterǁfit__mutmut_32': xǁBlackJaxFitterǁfit__mutmut_32, 
        'xǁBlackJaxFitterǁfit__mutmut_33': xǁBlackJaxFitterǁfit__mutmut_33, 
        'xǁBlackJaxFitterǁfit__mutmut_34': xǁBlackJaxFitterǁfit__mutmut_34, 
        'xǁBlackJaxFitterǁfit__mutmut_35': xǁBlackJaxFitterǁfit__mutmut_35, 
        'xǁBlackJaxFitterǁfit__mutmut_36': xǁBlackJaxFitterǁfit__mutmut_36, 
        'xǁBlackJaxFitterǁfit__mutmut_37': xǁBlackJaxFitterǁfit__mutmut_37, 
        'xǁBlackJaxFitterǁfit__mutmut_38': xǁBlackJaxFitterǁfit__mutmut_38, 
        'xǁBlackJaxFitterǁfit__mutmut_39': xǁBlackJaxFitterǁfit__mutmut_39, 
        'xǁBlackJaxFitterǁfit__mutmut_40': xǁBlackJaxFitterǁfit__mutmut_40, 
        'xǁBlackJaxFitterǁfit__mutmut_41': xǁBlackJaxFitterǁfit__mutmut_41, 
        'xǁBlackJaxFitterǁfit__mutmut_42': xǁBlackJaxFitterǁfit__mutmut_42, 
        'xǁBlackJaxFitterǁfit__mutmut_43': xǁBlackJaxFitterǁfit__mutmut_43, 
        'xǁBlackJaxFitterǁfit__mutmut_44': xǁBlackJaxFitterǁfit__mutmut_44, 
        'xǁBlackJaxFitterǁfit__mutmut_45': xǁBlackJaxFitterǁfit__mutmut_45, 
        'xǁBlackJaxFitterǁfit__mutmut_46': xǁBlackJaxFitterǁfit__mutmut_46, 
        'xǁBlackJaxFitterǁfit__mutmut_47': xǁBlackJaxFitterǁfit__mutmut_47
    }
    xǁBlackJaxFitterǁfit__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁfit'

    def get_parameter_estimates(self) -> dict[str, float]:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_orig(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_1(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is not None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_2(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError(None)

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_3(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("XXThe model has not been fitted yet.XX")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_4(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("the model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_5(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("THE MODEL HAS NOT BEEN FITTED YET.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_6(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = None

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_7(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = None

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_8(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate(None) for param in positions[0]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_9(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[1]}

        estimates = {param: float(jnp.mean(samples)) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_10(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = None
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_11(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(None) for param, samples in param_samples.items()}
        return estimates

    def xǁBlackJaxFitterǁget_parameter_estimates__mutmut_12(self) -> dict[str, float]:
        """Returns the mean of the posterior samples for each parameter."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Combine chains and extract positions
        positions = [state.position for state in self.states]

        # This is a simplification. For a real implementation, we would need to handle
        # the dictionary structure of the positions more carefully.
        # For now, assuming the positions are dictionaries of parameters.

        # Flatten the list of dictionaries
        param_samples = {param: jnp.concatenate([p[param] for p in positions]) for param in positions[0]}

        estimates = {param: float(jnp.mean(None)) for param, samples in param_samples.items()}
        return estimates
    
    xǁBlackJaxFitterǁget_parameter_estimates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_1': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_1, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_2': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_2, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_3': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_3, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_4': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_4, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_5': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_5, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_6': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_6, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_7': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_7, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_8': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_8, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_9': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_9, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_10': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_10, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_11': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_11, 
        'xǁBlackJaxFitterǁget_parameter_estimates__mutmut_12': xǁBlackJaxFitterǁget_parameter_estimates__mutmut_12
    }
    xǁBlackJaxFitterǁget_parameter_estimates__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁget_parameter_estimates'

    def _get_inference_data(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁ_get_inference_data__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁ_get_inference_data__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_orig(self):
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_1(self):
        if self.states is not None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_2(self):
        if self.states is None:
            raise RuntimeError(None)

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_3(self):
        if self.states is None:
            raise RuntimeError("XXThe model has not been fitted yet.XX")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_4(self):
        if self.states is None:
            raise RuntimeError("the model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_5(self):
        if self.states is None:
            raise RuntimeError("THE MODEL HAS NOT BEEN FITTED YET.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_6(self):
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = None

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_7(self):
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack(None) for param in self.states[0].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_8(self):
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[1].position
        }

        return az.from_dict(posterior_samples)

    def xǁBlackJaxFitterǁ_get_inference_data__mutmut_9(self):
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Assuming states is a list of states, one for each chain
        posterior_samples = {
            param: jnp.stack([chain.position[param] for chain in self.states]) for param in self.states[0].position
        }

        return az.from_dict(None)
    
    xǁBlackJaxFitterǁ_get_inference_data__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁ_get_inference_data__mutmut_1': xǁBlackJaxFitterǁ_get_inference_data__mutmut_1, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_2': xǁBlackJaxFitterǁ_get_inference_data__mutmut_2, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_3': xǁBlackJaxFitterǁ_get_inference_data__mutmut_3, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_4': xǁBlackJaxFitterǁ_get_inference_data__mutmut_4, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_5': xǁBlackJaxFitterǁ_get_inference_data__mutmut_5, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_6': xǁBlackJaxFitterǁ_get_inference_data__mutmut_6, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_7': xǁBlackJaxFitterǁ_get_inference_data__mutmut_7, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_8': xǁBlackJaxFitterǁ_get_inference_data__mutmut_8, 
        'xǁBlackJaxFitterǁ_get_inference_data__mutmut_9': xǁBlackJaxFitterǁ_get_inference_data__mutmut_9
    }
    xǁBlackJaxFitterǁ_get_inference_data__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁ_get_inference_data'

    def get_confidence_intervals(self) -> dict[str, tuple[float, float]]:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_orig(self) -> dict[str, tuple[float, float]]:
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

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_1(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is not None:
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

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_2(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError(None)

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

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_3(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("XXThe model has not been fitted yet.XX")

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

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_4(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("the model has not been fitted yet.")

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

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_5(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("THE MODEL HAS NOT BEEN FITTED YET.")

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

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_6(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = None
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_7(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = None

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_8(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(None, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_9(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=None)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_10(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_11(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, )

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_12(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=1.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_13(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = None
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_14(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(None),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_15(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "XXhdi_2.5%XX"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_16(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "HDI_2.5%"]),
                float(summary.loc[param, "hdi_97.5%"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_17(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(None),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_18(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "XXhdi_97.5%XX"]),
            )
            for param in summary.index
        }
        return intervals

    def xǁBlackJaxFitterǁget_confidence_intervals__mutmut_19(self) -> dict[str, tuple[float, float]]:
        """Returns the 95% confidence intervals for the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        summary = az.summary(idata, hdi_prob=0.95)

        intervals = {
            param: (
                float(summary.loc[param, "hdi_2.5%"]),
                float(summary.loc[param, "HDI_97.5%"]),
            )
            for param in summary.index
        }
        return intervals
    
    xǁBlackJaxFitterǁget_confidence_intervals__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_1': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_1, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_2': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_2, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_3': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_3, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_4': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_4, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_5': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_5, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_6': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_6, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_7': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_7, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_8': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_8, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_9': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_9, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_10': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_10, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_11': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_11, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_12': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_12, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_13': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_13, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_14': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_14, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_15': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_15, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_16': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_16, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_17': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_17, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_18': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_18, 
        'xǁBlackJaxFitterǁget_confidence_intervals__mutmut_19': xǁBlackJaxFitterǁget_confidence_intervals__mutmut_19
    }
    xǁBlackJaxFitterǁget_confidence_intervals__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁget_confidence_intervals'

    def get_summary(self) -> Any:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁget_summary__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁget_summary__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁget_summary__mutmut_orig(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_1(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is not None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_2(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError(None)

        idata = self._get_inference_data()
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_3(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError("XXThe model has not been fitted yet.XX")

        idata = self._get_inference_data()
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_4(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError("the model has not been fitted yet.")

        idata = self._get_inference_data()
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_5(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError("THE MODEL HAS NOT BEEN FITTED YET.")

        idata = self._get_inference_data()
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_6(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = None
        return az.summary(idata)

    def xǁBlackJaxFitterǁget_summary__mutmut_7(self) -> Any:
        """Returns a summary of the MCMC run using arviz."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        return az.summary(None)
    
    xǁBlackJaxFitterǁget_summary__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁget_summary__mutmut_1': xǁBlackJaxFitterǁget_summary__mutmut_1, 
        'xǁBlackJaxFitterǁget_summary__mutmut_2': xǁBlackJaxFitterǁget_summary__mutmut_2, 
        'xǁBlackJaxFitterǁget_summary__mutmut_3': xǁBlackJaxFitterǁget_summary__mutmut_3, 
        'xǁBlackJaxFitterǁget_summary__mutmut_4': xǁBlackJaxFitterǁget_summary__mutmut_4, 
        'xǁBlackJaxFitterǁget_summary__mutmut_5': xǁBlackJaxFitterǁget_summary__mutmut_5, 
        'xǁBlackJaxFitterǁget_summary__mutmut_6': xǁBlackJaxFitterǁget_summary__mutmut_6, 
        'xǁBlackJaxFitterǁget_summary__mutmut_7': xǁBlackJaxFitterǁget_summary__mutmut_7
    }
    xǁBlackJaxFitterǁget_summary__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁget_summary'

    def plot_trace(self, **kwargs) -> None:
        args = []# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁplot_trace__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁplot_trace__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁplot_trace__mutmut_orig(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_1(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is not None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_2(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError(None)

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_3(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("XXThe model has not been fitted yet.XX")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_4(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("the model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_5(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("THE MODEL HAS NOT BEEN FITTED YET.")

        idata = self._get_inference_data()
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_6(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = None
        az.plot_trace(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_7(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(None, **kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_8(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(**kwargs)

    def xǁBlackJaxFitterǁplot_trace__mutmut_9(self, **kwargs) -> None:
        """Plots the trace of the MCMC run."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_trace(idata, )
    
    xǁBlackJaxFitterǁplot_trace__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁplot_trace__mutmut_1': xǁBlackJaxFitterǁplot_trace__mutmut_1, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_2': xǁBlackJaxFitterǁplot_trace__mutmut_2, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_3': xǁBlackJaxFitterǁplot_trace__mutmut_3, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_4': xǁBlackJaxFitterǁplot_trace__mutmut_4, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_5': xǁBlackJaxFitterǁplot_trace__mutmut_5, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_6': xǁBlackJaxFitterǁplot_trace__mutmut_6, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_7': xǁBlackJaxFitterǁplot_trace__mutmut_7, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_8': xǁBlackJaxFitterǁplot_trace__mutmut_8, 
        'xǁBlackJaxFitterǁplot_trace__mutmut_9': xǁBlackJaxFitterǁplot_trace__mutmut_9
    }
    xǁBlackJaxFitterǁplot_trace__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁplot_trace'

    def plot_posterior(self, **kwargs) -> None:
        args = []# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBlackJaxFitterǁplot_posterior__mutmut_orig'), object.__getattribute__(self, 'xǁBlackJaxFitterǁplot_posterior__mutmut_mutants'), args, kwargs, self)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_orig(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_1(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is not None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_2(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError(None)

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_3(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("XXThe model has not been fitted yet.XX")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_4(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("the model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_5(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("THE MODEL HAS NOT BEEN FITTED YET.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_6(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = None
        az.plot_posterior(idata, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_7(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(None, **kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_8(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(**kwargs)

    def xǁBlackJaxFitterǁplot_posterior__mutmut_9(self, **kwargs) -> None:
        """Plots the posterior distributions of the parameters."""
        if self.states is None:
            raise RuntimeError("The model has not been fitted yet.")

        idata = self._get_inference_data()
        az.plot_posterior(idata, )
    
    xǁBlackJaxFitterǁplot_posterior__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBlackJaxFitterǁplot_posterior__mutmut_1': xǁBlackJaxFitterǁplot_posterior__mutmut_1, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_2': xǁBlackJaxFitterǁplot_posterior__mutmut_2, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_3': xǁBlackJaxFitterǁplot_posterior__mutmut_3, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_4': xǁBlackJaxFitterǁplot_posterior__mutmut_4, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_5': xǁBlackJaxFitterǁplot_posterior__mutmut_5, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_6': xǁBlackJaxFitterǁplot_posterior__mutmut_6, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_7': xǁBlackJaxFitterǁplot_posterior__mutmut_7, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_8': xǁBlackJaxFitterǁplot_posterior__mutmut_8, 
        'xǁBlackJaxFitterǁplot_posterior__mutmut_9': xǁBlackJaxFitterǁplot_posterior__mutmut_9
    }
    xǁBlackJaxFitterǁplot_posterior__mutmut_orig.__name__ = 'xǁBlackJaxFitterǁplot_posterior'
