from typing import Protocol, Type, Sequence, Dict
from abc import ABC, abstractmethod

# Define a type alias for array-like objects
ArrayLike = Sequence[float]

class Backend(Protocol):
    def array(self, data: ArrayLike) -> ArrayLike: ...
    def exp(self, x: ArrayLike) -> ArrayLike: ...
    def power(self, base: ArrayLike, exp: ArrayLike) -> ArrayLike: ...
    def solve_ode(self, f, y0: ArrayLike, t: ArrayLike) -> ArrayLike: ...
    def stack(self, arrays: Sequence[ArrayLike]) -> ArrayLike: ...
    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike: ...
    def zeros(self, shape: Sequence[int]) -> ArrayLike: ...
    def sum(self, x: ArrayLike) -> float: ...
    def mean(self, x: ArrayLike) -> float: ...
    def where(self, condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike: ...

# Default backend (will be set to NumPyBackend initially)
current_backend: Backend = None

def use_backend(backend_name: str):
    global current_backend
    if backend_name == "numpy":
        from .backends.numpy_backend import NumPyBackend
        current_backend = NumPyBackend()
    elif backend_name == "jax":
        from .backends.jax_backend import JaxBackend
        current_backend = JaxBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

# Initialize with NumPy backend by default
use_backend("numpy")
