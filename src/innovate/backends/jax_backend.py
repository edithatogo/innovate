from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve


class JaxBackend:
    def array(self, data):
        """Convert input data to a JAX array.

        Args:
            data: Input data of any type convertable to array.

        Returns
        -------
            A JAX array representation of the input data.
        """
        return jnp.asarray(data)

    def exp(self, x):
        """Calculate the exponential of all elements in the input array.

        Args:
            x: Input array or sequence.

        Returns
        -------
            Element-wise exponential of the input array.
        """
        return jnp.exp(x)

    def power(self, x, y):
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return jnp.power(x, y)

    def sum(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
        """Sum of array elements over a given axis.

        Args:
            a: Elements to sum.
            axis: Axis or axes along which to sum. None sums all elements.
            dtype: Type of the returned array and of the accumulator.
            keepdims: Whether to keep reduced dimensions.

        Returns
        -------
            Sum of elements, float if scalar, array if axis specified.
        """
        result = jnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def mean(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
        """Compute the arithmetic mean along the specified axis.

        Args:
            a: Array containing numbers whose mean is desired.
            axis: Axis or axes along which the means are computed.
                  None (default) computes the mean of the flattened array.
            dtype: Type to use in computing the mean.
            keepdims: Whether to keep reduced dimensions.

        Returns
        -------
            The mean of the elements, float if scalar, array if axis specified.
        """
        result = jnp.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def where(self, condition, x, y):
        return jnp.where(condition, x, y)

    def abs(self, x):
        return jnp.abs(x)

    def argmin(self, x):
        return jnp.argmin(x)

    def log(self, x):
        return jnp.log(x)

    def solve_ode(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = t[0]
        t1 = t[-1]
        saveat = SaveAt(ts=t)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def stack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.stack(arrays)

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    def zeros(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(shape)

    def max(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return jnp.max(x, axis=axis)

    def median(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return jnp.median(x, axis=axis)

    def interp(self, x, xp, fp):
        return jnp.interp(x, xp, fp)

    def jit(self, f: Callable) -> Callable:
        return jax.jit(f)

    def vmap(self, f: Callable) -> Callable:
        return jax.vmap(f)

    def clip(self, x, a_min, a_max):
        return jnp.clip(x, a_min, a_max)

    def ones(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.ones(shape)

    def zeros_like(self, x):
        return jnp.zeros_like(x)

    def ones_like(self, x):
        return jnp.ones_like(x)

    def ravel(self, x):
        return jnp.ravel(x)

    def gradient(self, x, *args, **kwargs):
        return jnp.gradient(x, *args, **kwargs)

    def min(self, x):
        return jnp.min(x)

    def copy(self, x):
        return jnp.copy(x)

    def diff(self, a, n=1, axis=-1):
        return jnp.diff(a, n=n, axis=axis)

    def sqrt(self, x):
        return jnp.sqrt(x)

    def vstack(self, x):
        return jnp.vstack(x)

    def logsumexp(self, x, axis=None):
        return jax.scipy.special.logsumexp(x, axis=axis)

    def polyfit(self, x, y, deg):
        return jnp.polyfit(x, y, deg)

    def lstsq(self, x, y, rcond):
        return jnp.linalg.lstsq(x, y, rcond=rcond)

    def nanmean(self, x):
        return jnp.nanmean(x)

    def isfinite(self, x):
        return jnp.isfinite(x)
