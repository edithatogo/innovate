import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
from typing import Sequence, Callable

class JaxBackend:
    def array(self, data):
        return jnp.asarray(data)

    exp = jnp.exp
    power = jnp.power
    sum = jnp.sum
    mean = jnp.mean
    where = jnp.where

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

    def jit(self, f: Callable) -> Callable:
        return jax.jit(f)

    def vmap(self, f: Callable) -> Callable:
        return jax.vmap(f)
