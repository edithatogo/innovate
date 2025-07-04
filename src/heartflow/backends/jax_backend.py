import jax.numpy as jnp
# from diffrax import diffeqsolve, ODETerm, Tsit5 # Uncomment in Phase 2
from typing import Sequence

class JaxBackend:
    array = jnp.ndarray
    exp = jnp.exp
    power = jnp.power

    def solve_ode(self, f, y0: Sequence[float], t: Sequence[float]) -> jnp.ndarray:
        # This will be implemented in Phase 2 using Diffrax
        raise NotImplementedError("JAX ODE solver not implemented in Phase 1. Will be available in Phase 2.")

    def stack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.stack(arrays)

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    def zeros(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(shape)
