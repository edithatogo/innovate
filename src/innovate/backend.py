from innovate.backends.numpy_backend import NumPyBackend

current_backend = NumPyBackend()

def use_backend(backend: str):
    global current_backend
    if backend == "jax":
        from innovate.backends.jax_backend import JaxBackend
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")

# Initialize with NumPy backend by default
use_backend("numpy")