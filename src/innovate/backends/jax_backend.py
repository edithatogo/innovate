import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
from typing import Sequence, Callable

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

    def solve_ode(self, f: Callable, y0: Sequence[float], t: Sequence[float]) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        saveat = SaveAt(ts=t)
        sol = diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[1] - t[0], y0=y0, saveat=saveat)
        return sol.ys

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
