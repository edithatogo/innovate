from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxopt import LBFGS

from innovate.backend import current_backend, use_backend
from innovate.base.base import DiffusionModel
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


class JaxFitter:
    """A fitter class that will use JAX for model estimation (Phase 2)."""

    def fit(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        args = [model, t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁJaxFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁJaxFitterǁfit__mutmut_orig(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_1(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = None
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_2(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(None)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_3(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = None

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_4(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(None)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_5(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = None
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_6(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend(None)

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_7(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("XXjaxXX")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_8(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("JAX")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_9(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = None

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_10(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(None)

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_11(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(None))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_12(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(None, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_13(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, None).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_14(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_15(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, ).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_16(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = None
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_17(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=None)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_18(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = None

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_19(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=None)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_20(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = None

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_21(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(None)

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_22(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(None, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_23(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, None))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_24(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_25(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, ))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_26(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            None,
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_27(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace(None, ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_28(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", None),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_29(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace(""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_30(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", ),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_31(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.upper().replace("backend", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_32(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("XXbackendXX", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_33(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("BACKEND", ""),
        )  # Restore original backend

        return model.params_

    def xǁJaxFitterǁfit__mutmut_34(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        original_backend = current_backend
        use_backend("jax")

        @jax.jit
        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model.cumulative_adoption(t_arr, *params_array)
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = jnp.array(list(model.initial_guesses(t, y).values()))

        opt = LBFGS(fun=loss_fn)
        sol = opt.run(init_params=initial_params)

        model.params_ = dict(zip(model.param_names, sol.params))

        use_backend(
            original_backend.__class__.__name__.lower().replace("backend", "XXXX"),
        )  # Restore original backend

        return model.params_
    
    xǁJaxFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxFitterǁfit__mutmut_1': xǁJaxFitterǁfit__mutmut_1, 
        'xǁJaxFitterǁfit__mutmut_2': xǁJaxFitterǁfit__mutmut_2, 
        'xǁJaxFitterǁfit__mutmut_3': xǁJaxFitterǁfit__mutmut_3, 
        'xǁJaxFitterǁfit__mutmut_4': xǁJaxFitterǁfit__mutmut_4, 
        'xǁJaxFitterǁfit__mutmut_5': xǁJaxFitterǁfit__mutmut_5, 
        'xǁJaxFitterǁfit__mutmut_6': xǁJaxFitterǁfit__mutmut_6, 
        'xǁJaxFitterǁfit__mutmut_7': xǁJaxFitterǁfit__mutmut_7, 
        'xǁJaxFitterǁfit__mutmut_8': xǁJaxFitterǁfit__mutmut_8, 
        'xǁJaxFitterǁfit__mutmut_9': xǁJaxFitterǁfit__mutmut_9, 
        'xǁJaxFitterǁfit__mutmut_10': xǁJaxFitterǁfit__mutmut_10, 
        'xǁJaxFitterǁfit__mutmut_11': xǁJaxFitterǁfit__mutmut_11, 
        'xǁJaxFitterǁfit__mutmut_12': xǁJaxFitterǁfit__mutmut_12, 
        'xǁJaxFitterǁfit__mutmut_13': xǁJaxFitterǁfit__mutmut_13, 
        'xǁJaxFitterǁfit__mutmut_14': xǁJaxFitterǁfit__mutmut_14, 
        'xǁJaxFitterǁfit__mutmut_15': xǁJaxFitterǁfit__mutmut_15, 
        'xǁJaxFitterǁfit__mutmut_16': xǁJaxFitterǁfit__mutmut_16, 
        'xǁJaxFitterǁfit__mutmut_17': xǁJaxFitterǁfit__mutmut_17, 
        'xǁJaxFitterǁfit__mutmut_18': xǁJaxFitterǁfit__mutmut_18, 
        'xǁJaxFitterǁfit__mutmut_19': xǁJaxFitterǁfit__mutmut_19, 
        'xǁJaxFitterǁfit__mutmut_20': xǁJaxFitterǁfit__mutmut_20, 
        'xǁJaxFitterǁfit__mutmut_21': xǁJaxFitterǁfit__mutmut_21, 
        'xǁJaxFitterǁfit__mutmut_22': xǁJaxFitterǁfit__mutmut_22, 
        'xǁJaxFitterǁfit__mutmut_23': xǁJaxFitterǁfit__mutmut_23, 
        'xǁJaxFitterǁfit__mutmut_24': xǁJaxFitterǁfit__mutmut_24, 
        'xǁJaxFitterǁfit__mutmut_25': xǁJaxFitterǁfit__mutmut_25, 
        'xǁJaxFitterǁfit__mutmut_26': xǁJaxFitterǁfit__mutmut_26, 
        'xǁJaxFitterǁfit__mutmut_27': xǁJaxFitterǁfit__mutmut_27, 
        'xǁJaxFitterǁfit__mutmut_28': xǁJaxFitterǁfit__mutmut_28, 
        'xǁJaxFitterǁfit__mutmut_29': xǁJaxFitterǁfit__mutmut_29, 
        'xǁJaxFitterǁfit__mutmut_30': xǁJaxFitterǁfit__mutmut_30, 
        'xǁJaxFitterǁfit__mutmut_31': xǁJaxFitterǁfit__mutmut_31, 
        'xǁJaxFitterǁfit__mutmut_32': xǁJaxFitterǁfit__mutmut_32, 
        'xǁJaxFitterǁfit__mutmut_33': xǁJaxFitterǁfit__mutmut_33, 
        'xǁJaxFitterǁfit__mutmut_34': xǁJaxFitterǁfit__mutmut_34
    }
    xǁJaxFitterǁfit__mutmut_orig.__name__ = 'xǁJaxFitterǁfit'
