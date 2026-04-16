"""Canonical runtime backend namespace for :mod:`innovate`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .numpy_backend import NumPyBackend


def _load_jax_backend() -> type[Any] | None:
    """Load the optional JAX backend class when the extra is installed."""
    try:
        from .jax_backend import JaxBackend
    except ImportError:  # pragma: no cover - optional dependency may be missing
        return None
    return JaxBackend


JaxBackend = _load_jax_backend()

__all__ = ["JaxBackend", "NumPyBackend", "current_backend", "use_backend"]


def __getattr__(name: str) -> Any:
    if name not in {"current_backend", "use_backend"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    backend_module = import_module("innovate.backend")
    return getattr(backend_module, name)
