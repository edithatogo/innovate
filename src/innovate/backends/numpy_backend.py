import numpy as np
from scipy.integrate import odeint
from typing import Sequence


class NumPyBackend:
    def array(self, data):
        return np.asarray(data)

    exp = np.exp
    power = np.power

    def sum(
        self,
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return np.sum(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def mean(self, a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return np.mean(
            a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        )

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def diff(self, a, n=1, axis=-1):
        return np.diff(a, n=n, axis=axis)

    def log(self, x):
        return np.log(x)

    def solve_ode(self, f, y0: Sequence[float], t: Sequence[float]) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        sol = odeint(f, y0, t)
        return sol

    def stack(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(arrays)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def zeros(self, shape: Sequence[int]) -> np.ndarray:
        return np.zeros(shape)

    def ones(self, shape: Sequence[int]) -> np.ndarray:
        return np.ones(shape)

    def max(self, x: np.ndarray) -> float:
        return np.max(x)

    def median(self, x: np.ndarray) -> float:
        return np.median(x)

    def interp(self, x, xp, fp):
        return np.interp(x, xp, fp)

    def jit(self, f):
        return f

    def vmap(self, f):
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(params, t_batched)])

        return mapped_f

    def zeros_like(self, x):
        return np.zeros_like(x)

    def ravel(self, x):
        return np.ravel(x)

    def argmin(self, x):
        return np.argmin(x)

    def abs(self, x):
        return np.abs(x)

    def gradient(self, x, *args, **kwargs):
        return np.gradient(x, *args, **kwargs)

    def clip(self, x, a_min, a_max):
        return np.clip(x, a_min, a_max)

    def min(self, x):
        return np.min(x)

    def copy(self, x):
        return np.copy(x)

    def vstack(self, x):
        return np.vstack(x)

    def polyfit(self, x, y, deg):
        return np.polyfit(x, y, deg)

    def lstsq(self, x, y, rcond):
        return np.linalg.lstsq(x, y, rcond=rcond)

    def nanmean(self, x):
        return np.nanmean(x)

    def isfinite(self, x):
        return np.isfinite(x)

    def errstate(self, **kwargs):
        return np.errstate(**kwargs)

    def sqrt(self, x):
        return np.sqrt(x)
