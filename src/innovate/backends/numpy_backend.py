from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import odeint
from scipy.special import logsumexp

if TYPE_CHECKING:
    import numpy.typing as npt

    ArrayLike = npt.ArrayLike
    NDArray = npt.NDArray


class NumPyBackend:
    def array(self, data: Any) -> np.ndarray:
        """Convert input data to a NumPy array.

        Args:
            data: Input data of any type convertable to array.

        Returns
        -------
            A NumPy array representation of the input data.
        """
        return np.asarray(data)

    def sum(
        self,
        a: np.ndarray | Sequence,
        axis: int | tuple | None = None,
        dtype: type | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        initial: float | None = None,
        where: np.ndarray | None = None,
    ) -> np.ndarray | float:
        """Sum of array elements over a given axis.

        Args:
            a: Elements to sum.
            axis: Axis or axes along which to sum. None sums all elements.
            dtype: Type of the returned array and of the accumulator.
            out: Alternative output array to place the result.
            keepdims: Whether to keep reduced dimensions.
            initial: Starting value for the sum.
            where: Elements to include in the sum.

        Returns
        -------
            Sum of elements, float if scalar, array if axis specified.
        """
        kwargs = {
            "axis": axis,
            "dtype": dtype,
            "out": out,
            "keepdims": keepdims,
        }
        if initial is not None:
            kwargs["initial"] = initial
        if where is not None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def mean(
        self,
        a: np.ndarray | Sequence,
        axis: int | tuple | None = None,
        dtype: type | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        *,
        where: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Compute the arithmetic mean along the specified axis.

        Args:
            a: Array containing numbers whose mean is desired.
            axis: Axis or axes along which the means are computed.
                  None (default) computes the mean of the flattened array.
            dtype: Type to use in computing the mean.
            out: Alternative output array to place the result.
            keepdims: Whether to keep reduced dimensions.
            where: Elements to include in the mean calculation.

        Returns
        -------
            The mean of the elements, float if scalar, array if axis specified.
        """
        kwargs = {
            "axis": axis,
            "dtype": dtype,
            "out": out,
            "keepdims": keepdims,
        }
        if where is not None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def where(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
        """Return elements chosen from x or y depending on condition.

        Args:
            condition: Where True, yield x, otherwise yield y.
            x: Values from which to choose when condition is True.
            y: Values from which to choose when condition is False.

        Returns
        -------
            An array with elements from x where condition is True, and elements
            from y elsewhere.
        """
        return np.where(condition, x, y)

    def diff(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=n, axis=axis)

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def logsumexp(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return logsumexp(x, axis=axis)

    def solve_ode(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
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

    def zeros(self, shape: int | Sequence[int]) -> np.ndarray:
        """Return a new array of given shape and type, filled with zeros.

        Args:
            shape: Shape of the new array, e.g., (2, 3) or 2.

        Returns
        -------
            Array of zeros with the specified shape.
        """
        return np.zeros(shape)

    def ones(self, shape: int | Sequence[int]) -> np.ndarray:
        """Return a new array of given shape and type, filled with ones.

        Args:
            shape: Shape of the new array, e.g., (2, 3) or 2.

        Returns
        -------
            Array of ones with the specified shape.
        """
        return np.ones(shape)

    def max(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return np.max(x, axis=axis)

    def median(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array or sequence containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return np.median(x, axis=axis)

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

    def ones_like(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def ravel(self, x: np.ndarray) -> np.ndarray:
        return np.ravel(x)

    def argmin(self, x: np.ndarray) -> int:
        return int(np.argmin(x))

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def gradient(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(x, *args, **kwargs)

    def clip(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, a_min, a_max)

    def min(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the minimum of an array or minimum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Minimum of the array elements, float if scalar, array if axis specified.
        """
        return np.min(x, axis=axis)

    def copy(self, x: np.ndarray) -> np.ndarray:
        return np.copy(x)

    def vstack(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return np.vstack(x)

    def polyfit(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, y, deg)

    def lstsq(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(x, y, rcond=rcond)

    def nanmean(self, x: np.ndarray) -> float:
        return float(np.nanmean(x))

    def isfinite(self, x: np.ndarray) -> np.ndarray:
        return np.isfinite(x)

    def errstate(self, **kwargs: Any) -> Any:
        return np.errstate(**kwargs)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def exp(self, x: np.ndarray | Sequence) -> np.ndarray:
        """Calculate the exponential of all elements in the input array.

        Args:
            x: Input array or sequence.

        Returns
        -------
            Element-wise exponential of the input array.
        """
        return np.exp(x)

    def any(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if any element evaluates to True, or array if axis specified.
        """
        return np.any(a, axis=axis)

    def all(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if all elements evaluate to True, or array if axis specified.
        """
        return np.all(a, axis=axis)

    def squeeze(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array.

        Args:
            a: Input array.
            axis: Selects subset of single-dimensional entries in the shape.
                  If None (default), squeezes all single-dimensional entries.

        Returns
        -------
            Squeezed array with specified dimensions removed.
        """
        return np.squeeze(a, axis=axis)

    def repeat(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
        """Repeat elements of an array.

        Args:
            a: Input array.
            repeats: Number of repetitions for each element.
                     If repeats is array-like, it must broadcast with a.
            axis: Axis along which to repeat values.
                  If None (default), flattened input is used.

        Returns
        -------
            Output array with repeated elements.
        """
        return np.repeat(a, repeats, axis=axis)

    def power(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return np.power(x, y)

    def empty_like(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return a new array with the same shape and type as a given array, without initializing entries.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array with same shape and type as input, without initialized values.
        """
        return np.empty_like(a, dtype=dtype, subok=subok, shape=shape)

    def full_like(
        self,
        a: np.ndarray | Sequence,
        fill_value: int | float,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return a full array with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            fill_value: Value to fill the output array with.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array with same shape and type as input, filled with fill_value.
        """
        return np.full_like(a, fill_value, dtype=dtype, subok=subok, shape=shape)
