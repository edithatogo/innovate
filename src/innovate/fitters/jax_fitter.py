import jax
import jax.numpy as jnp
from jaxopt import LBFGS
from typing import Sequence, Dict
from innovate.base.base import DiffusionModel
from innovate.backend import use_backend, current_backend


class JaxFitter:
    """A fitter class that will use JAX for model estimation (Phase 2)."""

    def fit(
        self, model: DiffusionModel, t: Sequence[float], y: Sequence[float], **kwargs
    ) -> Dict[str, float]:
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
            original_backend.__class__.__name__.lower().replace("backend", "")
        )  # Restore original backend

        return model.params_
