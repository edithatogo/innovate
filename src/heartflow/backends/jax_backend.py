import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5
from typing import Sequence

class JaxBackend:
    def array(self, data: Sequence[float]) -> jnp.ndarray:
        return jnp.asarray(data)

    def exp(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)

    def power(self, base: jnp.ndarray, exp: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(base, exp)

    def solve_ode(self, f, y0: Sequence[float], t: Sequence[float]) -> jnp.ndarray:
        term = ODETerm(f)
        sol = diffeqsolve(term, Tsit5(), t[0], t[-1], t, y0)
        return sol.ys

    def stack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.stack(arrays)

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    def zeros(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(shape)

    def sum(self, x: jnp.ndarray) -> float:
        return jnp.sum(x)

    def mean(self, x: jnp.ndarray) -> float:
        return jnp.mean(x)

    def where(self, condition: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(condition, x, y)

    def jit(self, f):
        from jax import jit
        return jit(f)
