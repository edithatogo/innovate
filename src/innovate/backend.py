from innovate.backends.numpy_backend import NumPyBackend
try:
    from innovate.backends.jax_backend import JaxBackend
except Exception:  # pragma: no cover - optional dependency may be missing
    JaxBackend = None

current_backend = NumPyBackend()

def use_backend(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError("JAX backend is not available")
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")

# Initialize with NumPy backend by default
use_backend("numpy")