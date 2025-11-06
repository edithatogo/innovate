from typing import Any, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
from scipy.integrate import odeint
from scipy.special import logsumexp

if TYPE_CHECKING:
    import numpy.typing as npt
    ArrayLike = npt.ArrayLike
    NDArray = npt.NDArray


class NumPyBackend:
    def array(self, data: Any) -> np.ndarray:
        return np.asarray(data)

    def sum(
        self,
        a: Union[np.ndarray, Sequence],
        axis: Optional[Union[int, tuple]] = None,
        dtype: Optional[type] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        initial: Optional[float] = None,
        where: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, float]:
        kwargs = {
            'axis': axis,
            'dtype': dtype,
            'out': out,
            'keepdims': keepdims,
        }
        if initial is not None:
            kwargs['initial'] = initial
        if where is not None:
            kwargs['where'] = where
        return np.sum(a, **kwargs)

    def mean(
        self,
        a: Union[np.ndarray, Sequence],
        axis: Optional[Union[int, tuple]] = None,
        dtype: Optional[type] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        *,
        where: Optional[np.ndarray] = None
    ) -> float:
        kwargs = {
            'axis': axis,
            'dtype': dtype,
            'out': out,
            'keepdims': keepdims,
        }
        if where is not None:
            kwargs['where'] = where
        result = np.mean(a, **kwargs)
        return float(result)

    def where(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
        return np.where(condition, x, y)

    def diff(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=n, axis=axis)

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def logsumexp(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return logsumexp(x, axis=axis)

    def solve_ode(
        self,
        f: Any,
        y0: Union[Sequence, np.ndarray],
        t: Union[Sequence, np.ndarray]
    ) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(f, y0_array, t_array)
        return sol

    def stack(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(arrays)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def zeros(self, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return np.zeros(shape)

    def ones(self, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return np.ones(shape)

    def max(self, x: Union[np.ndarray, Sequence]) -> float:
        return float(np.max(x))

    def median(self, x: Union[np.ndarray, Sequence]) -> float:
        return float(np.median(x))

    def interp(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(x, xp, fp)

    def jit(self, f: Any) -> Any:
        return f

    def vmap(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(params, t_batched)])

        return mapped_f

    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def ravel(self, x: np.ndarray) -> np.ndarray:
        return np.ravel(x)

    def argmin(self, x: np.ndarray) -> int:
        return int(np.argmin(x))

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def gradient(self, x: Union[np.ndarray, Sequence], *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(x, *args, **kwargs)

    def clip(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, a_min, a_max)

    def min(self, x: Union[np.ndarray, Sequence]) -> float:
        return float(np.min(x))

    def copy(self, x: np.ndarray) -> np.ndarray:
        return np.copy(x)

    def vstack(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return np.vstack(x)

    def polyfit(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, y, deg)

    def lstsq(self, x: np.ndarray, y: np.ndarray, rcond: Optional[float]) -> tuple:
        return np.linalg.lstsq(x, y, rcond=rcond)

    def nanmean(self, x: np.ndarray) -> float:
        return float(np.nanmean(x))

    def isfinite(self, x: np.ndarray) -> np.ndarray:
        return np.isfinite(x)

    def errstate(self, **kwargs: Any) -> Any:
        return np.errstate(**kwargs)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def exp(self, x: Union[np.ndarray, Sequence]) -> np.ndarray:
        return np.exp(x)

    def any(self, a: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> Union[bool, np.ndarray]:
        return np.any(a, axis=axis)

    def all(self, a: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> Union[bool, np.ndarray]:
        return np.all(a, axis=axis)

    def squeeze(self, a: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> np.ndarray:
        return np.squeeze(a, axis=axis)

    def repeat(self, a: np.ndarray, repeats: Union[int, Sequence], axis: Optional[int] = None) -> np.ndarray:
        return np.repeat(a, repeats, axis=axis)

    def ones_like(self, a: Union[np.ndarray, Sequence], dtype: Optional[type] = None, subok: bool = True, shape: Optional[Union[int, Sequence]] = None) -> np.ndarray:
        return np.ones_like(a, dtype=dtype, subok=subok, shape=shape)

    def zeros_like(self, a: Union[np.ndarray, Sequence], dtype: Optional[type] = None, subok: bool = True, shape: Optional[Union[int, Sequence]] = None) -> np.ndarray:
        return np.zeros_like(a, dtype=dtype, subok=subok, shape=shape)

    def empty_like(self, a: Union[np.ndarray, Sequence], dtype: Optional[type] = None, subok: bool = True, shape: Optional[Union[int, Sequence]] = None) -> np.ndarray:
        return np.empty_like(a, dtype=dtype, subok=subok, shape=shape)

    def full_like(self, a: Union[np.ndarray, Sequence], fill_value: Union[int, float], dtype: Optional[type] = None, subok: bool = True, shape: Optional[Union[int, Sequence]] = None) -> np.ndarray:
        return np.full_like(a, fill_value, dtype=dtype, subok=subok, shape=shape)
