from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve


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

    def abs(self, x):
        return jnp.abs(x)

    def argmin(self, x):
        return jnp.argmin(x)

    def log(self, x):
        return jnp.log(x)

    def solve_ode(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = t[0]
        t1 = t[-1]
        saveat = SaveAt(ts=t)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
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

    def clip(self, x, a_min, a_max):
        return jnp.clip(x, a_min, a_max)

    def ones(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.ones(shape)

    def zeros_like(self, x):
        return jnp.zeros_like(x)

    def ones_like(self, x):
        return jnp.ones_like(x)

    def ravel(self, x):
        return jnp.ravel(x)

    def gradient(self, x, *args, **kwargs):
        return jnp.gradient(x, *args, **kwargs)

    def min(self, x):
        return jnp.min(x)

    def copy(self, x):
        return jnp.copy(x)

    def diff(self, a, n=1, axis=-1):
        return jnp.diff(a, n=n, axis=axis)

    def sqrt(self, x):
        return jnp.sqrt(x)

    def vstack(self, x):
        return jnp.vstack(x)

    def logsumexp(self, x, axis=None):
        return jax.scipy.special.logsumexp(x, axis=axis)

    def polyfit(self, x, y, deg):
        return jnp.polyfit(x, y, deg)

    def lstsq(self, x, y, rcond):
        return jnp.linalg.lstsq(x, y, rcond=rcond)

    def nanmean(self, x):
        return jnp.nanmean(x)

    def isfinite(self, x):
        return jnp.isfinite(x)
