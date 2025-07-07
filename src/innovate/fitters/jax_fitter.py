import jax
import jax.numpy as jnp
from jaxopt import LBFGS
from typing import Sequence, Dict

class JaxFitter:
    """A fitter class that will use JAX for model estimation (Phase 2)."""

    def fit(self, model, t: Sequence[float], y: Sequence[float], **kwargs) -> Dict[str, float]:
        t_arr = jnp.asarray(t)
        y_arr = jnp.asarray(y)

        from heartflow.backend import current_backend, use_backend
        original_backend = current_backend
        use_backend("jax")

        def _logistic_cumulative(t, L, k, x0):
            from heartflow.backend import current_backend as B
            return L / (1 + B.exp(-k * (t - x0)))
        
        model._logistic_cumulative = _logistic_cumulative

        def loss_fn(params_array):
            # Temporarily set model parameters for prediction within the loss function
            predictions = model._logistic_cumulative(t_arr, params_array[0], params_array[1], params_array[2])
            return jnp.sum((y_arr - predictions) ** 2)

        initial_params = model._get_initial_params_as_array(t, y)

        opt = LBFGS(fun=loss_fn, **kwargs)
        sol = opt.run(init_params=initial_params)
        
        model._L, model._k, model._x0 = sol.params[0], sol.params[1], sol.params[2]

        use_backend(original_backend.__class__.__name__.lower().replace('backend', '')) # Restore original backend

        return model.params_
