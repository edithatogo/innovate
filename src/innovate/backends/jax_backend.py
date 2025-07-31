import jax
import jax.numpy as jnp
from typing import Sequence, Callable
from jax.experimental.ode import odeint as jax_odeint


class JaxBackend:
    def array(self, data):
        return jnp.asarray(data)

    def exp(self, x):
        return jnp.exp(x)

    def power(self, x, y):
        return jnp.power(x, y)

    def sum(self, a, axis=None, dtype=None, keepdims=False):
        return jnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)

    def mean(self, a, axis=None, dtype=None, keepdims=False):
        return jnp.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)

    def where(self, condition, x, y):
        return jnp.where(condition, x, y)

    def log(self, x):
        return jnp.log(x)

    def solve_ode(
        self, f: Callable, y0: Sequence[float], t: Sequence[float], args=None
    ) -> jnp.ndarray:
        if args is None:
            return jax_odeint(f, y0, t, rtol=1e-6, atol=1e-5, mxstep=1000)
        else:
            return jax_odeint(f, y0, t, args, rtol=1e-6, atol=1e-5, mxstep=1000)

    def stack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.stack(arrays)

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    def zeros(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(shape)

    def max(self, x: jnp.ndarray) -> float:
        return jnp.max(x)

    def median(self, x: jnp.ndarray) -> float:
        return jnp.median(x)

    def interp(self, x, xp, fp):
        return jnp.interp(x, xp, fp)

    def jit(self, f: Callable) -> Callable:
        return jax.jit(f)

    def vmap(self, f: Callable) -> Callable:
        return jax.vmap(f)
