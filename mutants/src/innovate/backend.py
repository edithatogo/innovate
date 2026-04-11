"""Backend selection for the :mod:`innovate` library."""

from innovate.backends.numpy_backend import NumPyBackend

# JAX and diffrax are optional dependencies
try:
    from innovate.backends.jax_backend import JaxBackend  # type: ignore
except ImportError:  # pragma: no cover - optional dependency may be missing
    JaxBackend = None

current_backend = NumPyBackend()
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


def use_backend(backend: str):
    args = [backend]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_use_backend__mutmut_orig, x_use_backend__mutmut_mutants, args, kwargs, None)


def x_use_backend__mutmut_orig(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_1(backend: str):
    global current_backend
    if backend != "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_2(backend: str):
    global current_backend
    if backend == "XXjaxXX":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_3(backend: str):
    global current_backend
    if backend == "JAX":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_4(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is not None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_5(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                None,
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_6(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "XXJAX backend is not available. Install jax and diffrax to use it.XX",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_7(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "jax backend is not available. install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_8(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX BACKEND IS NOT AVAILABLE. INSTALL JAX AND DIFFRAX TO USE IT.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_9(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = None
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_10(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend != "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_11(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "XXnumpyXX":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_12(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "NUMPY":
        current_backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_13(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = None
    else:
        raise ValueError(f"Unknown backend: {backend}")


def x_use_backend__mutmut_14(backend: str):
    global current_backend
    if backend == "jax":
        if JaxBackend is None:
            raise ImportError(
                "JAX backend is not available. Install jax and diffrax to use it.",
            )
        current_backend = JaxBackend()
    elif backend == "numpy":
        current_backend = NumPyBackend()
    else:
        raise ValueError(None)

x_use_backend__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_use_backend__mutmut_1': x_use_backend__mutmut_1, 
    'x_use_backend__mutmut_2': x_use_backend__mutmut_2, 
    'x_use_backend__mutmut_3': x_use_backend__mutmut_3, 
    'x_use_backend__mutmut_4': x_use_backend__mutmut_4, 
    'x_use_backend__mutmut_5': x_use_backend__mutmut_5, 
    'x_use_backend__mutmut_6': x_use_backend__mutmut_6, 
    'x_use_backend__mutmut_7': x_use_backend__mutmut_7, 
    'x_use_backend__mutmut_8': x_use_backend__mutmut_8, 
    'x_use_backend__mutmut_9': x_use_backend__mutmut_9, 
    'x_use_backend__mutmut_10': x_use_backend__mutmut_10, 
    'x_use_backend__mutmut_11': x_use_backend__mutmut_11, 
    'x_use_backend__mutmut_12': x_use_backend__mutmut_12, 
    'x_use_backend__mutmut_13': x_use_backend__mutmut_13, 
    'x_use_backend__mutmut_14': x_use_backend__mutmut_14
}
x_use_backend__mutmut_orig.__name__ = 'x_use_backend'


# Initialize with the NumPy backend by default
use_backend("numpy")
