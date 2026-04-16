"""Backend selection for the :mod:`innovate` library."""

from __future__ import annotations

from typing import Any

from innovate.backends.numpy_backend import NumPyBackend
from innovate.capabilities import get_backend_capability


def _load_jax_backend() -> type[Any] | None:
    """Load the optional JAX backend class when the extra is installed."""
    try:
        from innovate.backends.jax_backend import JaxBackend
    except ImportError:  # pragma: no cover - optional dependency may be missing
        return None
    return JaxBackend


JaxBackend = _load_jax_backend()

current_backend = NumPyBackend()


def use_backend(backend: str) -> None:
    global current_backend  # noqa: PLW0603
    if backend == "jax":
        capability = get_backend_capability("jax")
        if JaxBackend is None or not capability.available:
            raise ImportError(
                "JAX backend is not available. Install innovate[jax] to enable it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Initialize with the NumPy backend by default
use_backend("numpy")
