from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
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


class JaxBackend:
    def array(self, data):
        args = [data]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁarray__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁarray__mutmut_mutants'), args, kwargs, self)
    def xǁJaxBackendǁarray__mutmut_orig(self, data):
        """Convert input data to a JAX array.

        Args:
            data: Input data of any type convertable to array.

        Returns
        -------
            A JAX array representation of the input data.
        """
        return jnp.asarray(data)
    def xǁJaxBackendǁarray__mutmut_1(self, data):
        """Convert input data to a JAX array.

        Args:
            data: Input data of any type convertable to array.

        Returns
        -------
            A JAX array representation of the input data.
        """
        return jnp.asarray(None)
    
    xǁJaxBackendǁarray__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁarray__mutmut_1': xǁJaxBackendǁarray__mutmut_1
    }
    xǁJaxBackendǁarray__mutmut_orig.__name__ = 'xǁJaxBackendǁarray'

    def exp(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁexp__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁexp__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁexp__mutmut_orig(self, x):
        """Calculate the exponential of all elements in the input array.

        Args:
            x: Input array or sequence.

        Returns
        -------
            Element-wise exponential of the input array.
        """
        return jnp.exp(x)

    def xǁJaxBackendǁexp__mutmut_1(self, x):
        """Calculate the exponential of all elements in the input array.

        Args:
            x: Input array or sequence.

        Returns
        -------
            Element-wise exponential of the input array.
        """
        return jnp.exp(None)
    
    xǁJaxBackendǁexp__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁexp__mutmut_1': xǁJaxBackendǁexp__mutmut_1
    }
    xǁJaxBackendǁexp__mutmut_orig.__name__ = 'xǁJaxBackendǁexp'

    def power(self, x, y):
        args = [x, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁpower__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁpower__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁpower__mutmut_orig(self, x, y):
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return jnp.power(x, y)

    def xǁJaxBackendǁpower__mutmut_1(self, x, y):
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return jnp.power(None, y)

    def xǁJaxBackendǁpower__mutmut_2(self, x, y):
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return jnp.power(x, None)

    def xǁJaxBackendǁpower__mutmut_3(self, x, y):
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return jnp.power(y)

    def xǁJaxBackendǁpower__mutmut_4(self, x, y):
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return jnp.power(x, )
    
    xǁJaxBackendǁpower__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁpower__mutmut_1': xǁJaxBackendǁpower__mutmut_1, 
        'xǁJaxBackendǁpower__mutmut_2': xǁJaxBackendǁpower__mutmut_2, 
        'xǁJaxBackendǁpower__mutmut_3': xǁJaxBackendǁpower__mutmut_3, 
        'xǁJaxBackendǁpower__mutmut_4': xǁJaxBackendǁpower__mutmut_4
    }
    xǁJaxBackendǁpower__mutmut_orig.__name__ = 'xǁJaxBackendǁpower'

    def sum(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
        args = [a, axis, dtype, keepdims]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁsum__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁsum__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁsum__mutmut_orig(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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

    def xǁJaxBackendǁsum__mutmut_1(self, a, axis=None, dtype=None, keepdims=True) -> float | jnp.ndarray:
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

    def xǁJaxBackendǁsum__mutmut_2(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = None
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_3(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(None, axis=axis, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_4(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(a, axis=None, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_5(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(a, axis=axis, dtype=None, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_6(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(a, axis=axis, dtype=dtype, keepdims=None)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_7(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_8(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(a, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_9(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(a, axis=axis, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_10(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.sum(a, axis=axis, dtype=dtype, )
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_11(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        if axis is None or not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_12(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        if axis is not None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_13(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        if axis is None and keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁsum__mutmut_14(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
            return float(None)
        return result
    
    xǁJaxBackendǁsum__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁsum__mutmut_1': xǁJaxBackendǁsum__mutmut_1, 
        'xǁJaxBackendǁsum__mutmut_2': xǁJaxBackendǁsum__mutmut_2, 
        'xǁJaxBackendǁsum__mutmut_3': xǁJaxBackendǁsum__mutmut_3, 
        'xǁJaxBackendǁsum__mutmut_4': xǁJaxBackendǁsum__mutmut_4, 
        'xǁJaxBackendǁsum__mutmut_5': xǁJaxBackendǁsum__mutmut_5, 
        'xǁJaxBackendǁsum__mutmut_6': xǁJaxBackendǁsum__mutmut_6, 
        'xǁJaxBackendǁsum__mutmut_7': xǁJaxBackendǁsum__mutmut_7, 
        'xǁJaxBackendǁsum__mutmut_8': xǁJaxBackendǁsum__mutmut_8, 
        'xǁJaxBackendǁsum__mutmut_9': xǁJaxBackendǁsum__mutmut_9, 
        'xǁJaxBackendǁsum__mutmut_10': xǁJaxBackendǁsum__mutmut_10, 
        'xǁJaxBackendǁsum__mutmut_11': xǁJaxBackendǁsum__mutmut_11, 
        'xǁJaxBackendǁsum__mutmut_12': xǁJaxBackendǁsum__mutmut_12, 
        'xǁJaxBackendǁsum__mutmut_13': xǁJaxBackendǁsum__mutmut_13, 
        'xǁJaxBackendǁsum__mutmut_14': xǁJaxBackendǁsum__mutmut_14
    }
    xǁJaxBackendǁsum__mutmut_orig.__name__ = 'xǁJaxBackendǁsum'

    def mean(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
        args = [a, axis, dtype, keepdims]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁmean__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁmean__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁmean__mutmut_orig(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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

    def xǁJaxBackendǁmean__mutmut_1(self, a, axis=None, dtype=None, keepdims=True) -> float | jnp.ndarray:
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

    def xǁJaxBackendǁmean__mutmut_2(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = None
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_3(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(None, axis=axis, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_4(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(a, axis=None, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_5(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(a, axis=axis, dtype=None, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_6(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(a, axis=axis, dtype=dtype, keepdims=None)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_7(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(axis=axis, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_8(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(a, dtype=dtype, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_9(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(a, axis=axis, keepdims=keepdims)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_10(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        result = jnp.mean(a, axis=axis, dtype=dtype, )
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_11(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        if axis is None or not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_12(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        if axis is not None and not keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_13(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
        if axis is None and keepdims:
            return float(result)
        return result

    def xǁJaxBackendǁmean__mutmut_14(self, a, axis=None, dtype=None, keepdims=False) -> float | jnp.ndarray:
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
            return float(None)
        return result
    
    xǁJaxBackendǁmean__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁmean__mutmut_1': xǁJaxBackendǁmean__mutmut_1, 
        'xǁJaxBackendǁmean__mutmut_2': xǁJaxBackendǁmean__mutmut_2, 
        'xǁJaxBackendǁmean__mutmut_3': xǁJaxBackendǁmean__mutmut_3, 
        'xǁJaxBackendǁmean__mutmut_4': xǁJaxBackendǁmean__mutmut_4, 
        'xǁJaxBackendǁmean__mutmut_5': xǁJaxBackendǁmean__mutmut_5, 
        'xǁJaxBackendǁmean__mutmut_6': xǁJaxBackendǁmean__mutmut_6, 
        'xǁJaxBackendǁmean__mutmut_7': xǁJaxBackendǁmean__mutmut_7, 
        'xǁJaxBackendǁmean__mutmut_8': xǁJaxBackendǁmean__mutmut_8, 
        'xǁJaxBackendǁmean__mutmut_9': xǁJaxBackendǁmean__mutmut_9, 
        'xǁJaxBackendǁmean__mutmut_10': xǁJaxBackendǁmean__mutmut_10, 
        'xǁJaxBackendǁmean__mutmut_11': xǁJaxBackendǁmean__mutmut_11, 
        'xǁJaxBackendǁmean__mutmut_12': xǁJaxBackendǁmean__mutmut_12, 
        'xǁJaxBackendǁmean__mutmut_13': xǁJaxBackendǁmean__mutmut_13, 
        'xǁJaxBackendǁmean__mutmut_14': xǁJaxBackendǁmean__mutmut_14
    }
    xǁJaxBackendǁmean__mutmut_orig.__name__ = 'xǁJaxBackendǁmean'

    def where(self, condition, x, y):
        args = [condition, x, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁwhere__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁwhere__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁwhere__mutmut_orig(self, condition, x, y):
        return jnp.where(condition, x, y)

    def xǁJaxBackendǁwhere__mutmut_1(self, condition, x, y):
        return jnp.where(None, x, y)

    def xǁJaxBackendǁwhere__mutmut_2(self, condition, x, y):
        return jnp.where(condition, None, y)

    def xǁJaxBackendǁwhere__mutmut_3(self, condition, x, y):
        return jnp.where(condition, x, None)

    def xǁJaxBackendǁwhere__mutmut_4(self, condition, x, y):
        return jnp.where(x, y)

    def xǁJaxBackendǁwhere__mutmut_5(self, condition, x, y):
        return jnp.where(condition, y)

    def xǁJaxBackendǁwhere__mutmut_6(self, condition, x, y):
        return jnp.where(condition, x, )
    
    xǁJaxBackendǁwhere__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁwhere__mutmut_1': xǁJaxBackendǁwhere__mutmut_1, 
        'xǁJaxBackendǁwhere__mutmut_2': xǁJaxBackendǁwhere__mutmut_2, 
        'xǁJaxBackendǁwhere__mutmut_3': xǁJaxBackendǁwhere__mutmut_3, 
        'xǁJaxBackendǁwhere__mutmut_4': xǁJaxBackendǁwhere__mutmut_4, 
        'xǁJaxBackendǁwhere__mutmut_5': xǁJaxBackendǁwhere__mutmut_5, 
        'xǁJaxBackendǁwhere__mutmut_6': xǁJaxBackendǁwhere__mutmut_6
    }
    xǁJaxBackendǁwhere__mutmut_orig.__name__ = 'xǁJaxBackendǁwhere'

    def abs(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁabs__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁabs__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁabs__mutmut_orig(self, x):
        return jnp.abs(x)

    def xǁJaxBackendǁabs__mutmut_1(self, x):
        return jnp.abs(None)
    
    xǁJaxBackendǁabs__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁabs__mutmut_1': xǁJaxBackendǁabs__mutmut_1
    }
    xǁJaxBackendǁabs__mutmut_orig.__name__ = 'xǁJaxBackendǁabs'

    def argmin(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁargmin__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁargmin__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁargmin__mutmut_orig(self, x):
        return jnp.argmin(x)

    def xǁJaxBackendǁargmin__mutmut_1(self, x):
        return jnp.argmin(None)
    
    xǁJaxBackendǁargmin__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁargmin__mutmut_1': xǁJaxBackendǁargmin__mutmut_1
    }
    xǁJaxBackendǁargmin__mutmut_orig.__name__ = 'xǁJaxBackendǁargmin'

    def log(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁlog__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁlog__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁlog__mutmut_orig(self, x):
        return jnp.log(x)

    def xǁJaxBackendǁlog__mutmut_1(self, x):
        return jnp.log(None)
    
    xǁJaxBackendǁlog__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁlog__mutmut_1': xǁJaxBackendǁlog__mutmut_1
    }
    xǁJaxBackendǁlog__mutmut_orig.__name__ = 'xǁJaxBackendǁlog'

    def solve_ode(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        args = [f, y0, t, args]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁsolve_ode__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁsolve_ode__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁsolve_ode__mutmut_orig(
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

    def xǁJaxBackendǁsolve_ode__mutmut_1(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = None
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

    def xǁJaxBackendǁsolve_ode__mutmut_2(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(None)
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

    def xǁJaxBackendǁsolve_ode__mutmut_3(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = None
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

    def xǁJaxBackendǁsolve_ode__mutmut_4(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = None
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

    def xǁJaxBackendǁsolve_ode__mutmut_5(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = t[1]
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

    def xǁJaxBackendǁsolve_ode__mutmut_6(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = t[0]
        t1 = None
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

    def xǁJaxBackendǁsolve_ode__mutmut_7(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = t[0]
        t1 = t[+1]
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

    def xǁJaxBackendǁsolve_ode__mutmut_8(
        self,
        f: Callable,
        y0: Sequence[float],
        t: Sequence[float],
        args=None,
    ) -> jnp.ndarray:
        term = ODETerm(f)
        solver = Dopri5()
        t0 = t[0]
        t1 = t[-2]
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

    def xǁJaxBackendǁsolve_ode__mutmut_9(
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
        saveat = None
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

    def xǁJaxBackendǁsolve_ode__mutmut_10(
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
        saveat = SaveAt(ts=None)
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

    def xǁJaxBackendǁsolve_ode__mutmut_11(
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
        sol = None
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_12(
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
            None,
            solver,
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_13(
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
            None,
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_14(
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
            None,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_15(
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
            None,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_16(
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
            dt0=None,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_17(
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
            y0=None,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_18(
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
            saveat=None,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_19(
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
            args=None,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_20(
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
            solver,
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_21(
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
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_22(
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
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_23(
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
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_24(
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
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_25(
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
            saveat=saveat,
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_26(
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
            args=args,
        )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_27(
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
            )
        return sol.ys

    def xǁJaxBackendǁsolve_ode__mutmut_28(
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
            dt0=1.1,
            y0=y0,
            saveat=saveat,
            args=args,
        )
        return sol.ys
    
    xǁJaxBackendǁsolve_ode__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁsolve_ode__mutmut_1': xǁJaxBackendǁsolve_ode__mutmut_1, 
        'xǁJaxBackendǁsolve_ode__mutmut_2': xǁJaxBackendǁsolve_ode__mutmut_2, 
        'xǁJaxBackendǁsolve_ode__mutmut_3': xǁJaxBackendǁsolve_ode__mutmut_3, 
        'xǁJaxBackendǁsolve_ode__mutmut_4': xǁJaxBackendǁsolve_ode__mutmut_4, 
        'xǁJaxBackendǁsolve_ode__mutmut_5': xǁJaxBackendǁsolve_ode__mutmut_5, 
        'xǁJaxBackendǁsolve_ode__mutmut_6': xǁJaxBackendǁsolve_ode__mutmut_6, 
        'xǁJaxBackendǁsolve_ode__mutmut_7': xǁJaxBackendǁsolve_ode__mutmut_7, 
        'xǁJaxBackendǁsolve_ode__mutmut_8': xǁJaxBackendǁsolve_ode__mutmut_8, 
        'xǁJaxBackendǁsolve_ode__mutmut_9': xǁJaxBackendǁsolve_ode__mutmut_9, 
        'xǁJaxBackendǁsolve_ode__mutmut_10': xǁJaxBackendǁsolve_ode__mutmut_10, 
        'xǁJaxBackendǁsolve_ode__mutmut_11': xǁJaxBackendǁsolve_ode__mutmut_11, 
        'xǁJaxBackendǁsolve_ode__mutmut_12': xǁJaxBackendǁsolve_ode__mutmut_12, 
        'xǁJaxBackendǁsolve_ode__mutmut_13': xǁJaxBackendǁsolve_ode__mutmut_13, 
        'xǁJaxBackendǁsolve_ode__mutmut_14': xǁJaxBackendǁsolve_ode__mutmut_14, 
        'xǁJaxBackendǁsolve_ode__mutmut_15': xǁJaxBackendǁsolve_ode__mutmut_15, 
        'xǁJaxBackendǁsolve_ode__mutmut_16': xǁJaxBackendǁsolve_ode__mutmut_16, 
        'xǁJaxBackendǁsolve_ode__mutmut_17': xǁJaxBackendǁsolve_ode__mutmut_17, 
        'xǁJaxBackendǁsolve_ode__mutmut_18': xǁJaxBackendǁsolve_ode__mutmut_18, 
        'xǁJaxBackendǁsolve_ode__mutmut_19': xǁJaxBackendǁsolve_ode__mutmut_19, 
        'xǁJaxBackendǁsolve_ode__mutmut_20': xǁJaxBackendǁsolve_ode__mutmut_20, 
        'xǁJaxBackendǁsolve_ode__mutmut_21': xǁJaxBackendǁsolve_ode__mutmut_21, 
        'xǁJaxBackendǁsolve_ode__mutmut_22': xǁJaxBackendǁsolve_ode__mutmut_22, 
        'xǁJaxBackendǁsolve_ode__mutmut_23': xǁJaxBackendǁsolve_ode__mutmut_23, 
        'xǁJaxBackendǁsolve_ode__mutmut_24': xǁJaxBackendǁsolve_ode__mutmut_24, 
        'xǁJaxBackendǁsolve_ode__mutmut_25': xǁJaxBackendǁsolve_ode__mutmut_25, 
        'xǁJaxBackendǁsolve_ode__mutmut_26': xǁJaxBackendǁsolve_ode__mutmut_26, 
        'xǁJaxBackendǁsolve_ode__mutmut_27': xǁJaxBackendǁsolve_ode__mutmut_27, 
        'xǁJaxBackendǁsolve_ode__mutmut_28': xǁJaxBackendǁsolve_ode__mutmut_28
    }
    xǁJaxBackendǁsolve_ode__mutmut_orig.__name__ = 'xǁJaxBackendǁsolve_ode'

    def stack(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        args = [arrays]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁstack__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁstack__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁstack__mutmut_orig(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.stack(arrays)

    def xǁJaxBackendǁstack__mutmut_1(self, arrays: Sequence[jnp.ndarray]) -> jnp.ndarray:
        return jnp.stack(None)
    
    xǁJaxBackendǁstack__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁstack__mutmut_1': xǁJaxBackendǁstack__mutmut_1
    }
    xǁJaxBackendǁstack__mutmut_orig.__name__ = 'xǁJaxBackendǁstack'

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        args = [a, b]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁmatmul__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁmatmul__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁmatmul__mutmut_orig(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    def xǁJaxBackendǁmatmul__mutmut_1(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(None, b)

    def xǁJaxBackendǁmatmul__mutmut_2(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, None)

    def xǁJaxBackendǁmatmul__mutmut_3(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(b)

    def xǁJaxBackendǁmatmul__mutmut_4(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, )
    
    xǁJaxBackendǁmatmul__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁmatmul__mutmut_1': xǁJaxBackendǁmatmul__mutmut_1, 
        'xǁJaxBackendǁmatmul__mutmut_2': xǁJaxBackendǁmatmul__mutmut_2, 
        'xǁJaxBackendǁmatmul__mutmut_3': xǁJaxBackendǁmatmul__mutmut_3, 
        'xǁJaxBackendǁmatmul__mutmut_4': xǁJaxBackendǁmatmul__mutmut_4
    }
    xǁJaxBackendǁmatmul__mutmut_orig.__name__ = 'xǁJaxBackendǁmatmul'

    def zeros(self, shape: Sequence[int]) -> jnp.ndarray:
        args = [shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁzeros__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁzeros__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁzeros__mutmut_orig(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(shape)

    def xǁJaxBackendǁzeros__mutmut_1(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.zeros(None)
    
    xǁJaxBackendǁzeros__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁzeros__mutmut_1': xǁJaxBackendǁzeros__mutmut_1
    }
    xǁJaxBackendǁzeros__mutmut_orig.__name__ = 'xǁJaxBackendǁzeros'

    def max(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁmax__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁmax__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁmax__mutmut_orig(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return jnp.max(x, axis=axis)

    def xǁJaxBackendǁmax__mutmut_1(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return jnp.max(None, axis=axis)

    def xǁJaxBackendǁmax__mutmut_2(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return jnp.max(x, axis=None)

    def xǁJaxBackendǁmax__mutmut_3(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return jnp.max(axis=axis)

    def xǁJaxBackendǁmax__mutmut_4(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return jnp.max(x, )
    
    xǁJaxBackendǁmax__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁmax__mutmut_1': xǁJaxBackendǁmax__mutmut_1, 
        'xǁJaxBackendǁmax__mutmut_2': xǁJaxBackendǁmax__mutmut_2, 
        'xǁJaxBackendǁmax__mutmut_3': xǁJaxBackendǁmax__mutmut_3, 
        'xǁJaxBackendǁmax__mutmut_4': xǁJaxBackendǁmax__mutmut_4
    }
    xǁJaxBackendǁmax__mutmut_orig.__name__ = 'xǁJaxBackendǁmax'

    def median(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁmedian__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁmedian__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁmedian__mutmut_orig(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
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

    def xǁJaxBackendǁmedian__mutmut_1(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return jnp.median(None, axis=axis)

    def xǁJaxBackendǁmedian__mutmut_2(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return jnp.median(x, axis=None)

    def xǁJaxBackendǁmedian__mutmut_3(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return jnp.median(axis=axis)

    def xǁJaxBackendǁmedian__mutmut_4(self, x: jnp.ndarray, axis: int | tuple | None = None) -> float | jnp.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return jnp.median(x, )
    
    xǁJaxBackendǁmedian__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁmedian__mutmut_1': xǁJaxBackendǁmedian__mutmut_1, 
        'xǁJaxBackendǁmedian__mutmut_2': xǁJaxBackendǁmedian__mutmut_2, 
        'xǁJaxBackendǁmedian__mutmut_3': xǁJaxBackendǁmedian__mutmut_3, 
        'xǁJaxBackendǁmedian__mutmut_4': xǁJaxBackendǁmedian__mutmut_4
    }
    xǁJaxBackendǁmedian__mutmut_orig.__name__ = 'xǁJaxBackendǁmedian'

    def interp(self, x, xp, fp):
        args = [x, xp, fp]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁinterp__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁinterp__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁinterp__mutmut_orig(self, x, xp, fp):
        return jnp.interp(x, xp, fp)

    def xǁJaxBackendǁinterp__mutmut_1(self, x, xp, fp):
        return jnp.interp(None, xp, fp)

    def xǁJaxBackendǁinterp__mutmut_2(self, x, xp, fp):
        return jnp.interp(x, None, fp)

    def xǁJaxBackendǁinterp__mutmut_3(self, x, xp, fp):
        return jnp.interp(x, xp, None)

    def xǁJaxBackendǁinterp__mutmut_4(self, x, xp, fp):
        return jnp.interp(xp, fp)

    def xǁJaxBackendǁinterp__mutmut_5(self, x, xp, fp):
        return jnp.interp(x, fp)

    def xǁJaxBackendǁinterp__mutmut_6(self, x, xp, fp):
        return jnp.interp(x, xp, )
    
    xǁJaxBackendǁinterp__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁinterp__mutmut_1': xǁJaxBackendǁinterp__mutmut_1, 
        'xǁJaxBackendǁinterp__mutmut_2': xǁJaxBackendǁinterp__mutmut_2, 
        'xǁJaxBackendǁinterp__mutmut_3': xǁJaxBackendǁinterp__mutmut_3, 
        'xǁJaxBackendǁinterp__mutmut_4': xǁJaxBackendǁinterp__mutmut_4, 
        'xǁJaxBackendǁinterp__mutmut_5': xǁJaxBackendǁinterp__mutmut_5, 
        'xǁJaxBackendǁinterp__mutmut_6': xǁJaxBackendǁinterp__mutmut_6
    }
    xǁJaxBackendǁinterp__mutmut_orig.__name__ = 'xǁJaxBackendǁinterp'

    def jit(self, f: Callable) -> Callable:
        args = [f]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁjit__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁjit__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁjit__mutmut_orig(self, f: Callable) -> Callable:
        return jax.jit(f)

    def xǁJaxBackendǁjit__mutmut_1(self, f: Callable) -> Callable:
        return jax.jit(None)
    
    xǁJaxBackendǁjit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁjit__mutmut_1': xǁJaxBackendǁjit__mutmut_1
    }
    xǁJaxBackendǁjit__mutmut_orig.__name__ = 'xǁJaxBackendǁjit'

    def vmap(self, f: Callable) -> Callable:
        args = [f]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁvmap__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁvmap__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁvmap__mutmut_orig(self, f: Callable) -> Callable:
        return jax.vmap(f)

    def xǁJaxBackendǁvmap__mutmut_1(self, f: Callable) -> Callable:
        return jax.vmap(None)
    
    xǁJaxBackendǁvmap__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁvmap__mutmut_1': xǁJaxBackendǁvmap__mutmut_1
    }
    xǁJaxBackendǁvmap__mutmut_orig.__name__ = 'xǁJaxBackendǁvmap'

    def clip(self, x, a_min, a_max):
        args = [x, a_min, a_max]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁclip__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁclip__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁclip__mutmut_orig(self, x, a_min, a_max):
        return jnp.clip(x, a_min, a_max)

    def xǁJaxBackendǁclip__mutmut_1(self, x, a_min, a_max):
        return jnp.clip(None, a_min, a_max)

    def xǁJaxBackendǁclip__mutmut_2(self, x, a_min, a_max):
        return jnp.clip(x, None, a_max)

    def xǁJaxBackendǁclip__mutmut_3(self, x, a_min, a_max):
        return jnp.clip(x, a_min, None)

    def xǁJaxBackendǁclip__mutmut_4(self, x, a_min, a_max):
        return jnp.clip(a_min, a_max)

    def xǁJaxBackendǁclip__mutmut_5(self, x, a_min, a_max):
        return jnp.clip(x, a_max)

    def xǁJaxBackendǁclip__mutmut_6(self, x, a_min, a_max):
        return jnp.clip(x, a_min, )
    
    xǁJaxBackendǁclip__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁclip__mutmut_1': xǁJaxBackendǁclip__mutmut_1, 
        'xǁJaxBackendǁclip__mutmut_2': xǁJaxBackendǁclip__mutmut_2, 
        'xǁJaxBackendǁclip__mutmut_3': xǁJaxBackendǁclip__mutmut_3, 
        'xǁJaxBackendǁclip__mutmut_4': xǁJaxBackendǁclip__mutmut_4, 
        'xǁJaxBackendǁclip__mutmut_5': xǁJaxBackendǁclip__mutmut_5, 
        'xǁJaxBackendǁclip__mutmut_6': xǁJaxBackendǁclip__mutmut_6
    }
    xǁJaxBackendǁclip__mutmut_orig.__name__ = 'xǁJaxBackendǁclip'

    def ones(self, shape: Sequence[int]) -> jnp.ndarray:
        args = [shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁones__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁones__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁones__mutmut_orig(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.ones(shape)

    def xǁJaxBackendǁones__mutmut_1(self, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.ones(None)
    
    xǁJaxBackendǁones__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁones__mutmut_1': xǁJaxBackendǁones__mutmut_1
    }
    xǁJaxBackendǁones__mutmut_orig.__name__ = 'xǁJaxBackendǁones'

    def zeros_like(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁzeros_like__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁzeros_like__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁzeros_like__mutmut_orig(self, x):
        return jnp.zeros_like(x)

    def xǁJaxBackendǁzeros_like__mutmut_1(self, x):
        return jnp.zeros_like(None)
    
    xǁJaxBackendǁzeros_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁzeros_like__mutmut_1': xǁJaxBackendǁzeros_like__mutmut_1
    }
    xǁJaxBackendǁzeros_like__mutmut_orig.__name__ = 'xǁJaxBackendǁzeros_like'

    def ones_like(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁones_like__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁones_like__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁones_like__mutmut_orig(self, x):
        return jnp.ones_like(x)

    def xǁJaxBackendǁones_like__mutmut_1(self, x):
        return jnp.ones_like(None)
    
    xǁJaxBackendǁones_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁones_like__mutmut_1': xǁJaxBackendǁones_like__mutmut_1
    }
    xǁJaxBackendǁones_like__mutmut_orig.__name__ = 'xǁJaxBackendǁones_like'

    def ravel(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁravel__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁravel__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁravel__mutmut_orig(self, x):
        return jnp.ravel(x)

    def xǁJaxBackendǁravel__mutmut_1(self, x):
        return jnp.ravel(None)
    
    xǁJaxBackendǁravel__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁravel__mutmut_1': xǁJaxBackendǁravel__mutmut_1
    }
    xǁJaxBackendǁravel__mutmut_orig.__name__ = 'xǁJaxBackendǁravel'

    def gradient(self, x, *args, **kwargs):
        args = [x, *args]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁgradient__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁgradient__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁgradient__mutmut_orig(self, x, *args, **kwargs):
        return jnp.gradient(x, *args, **kwargs)

    def xǁJaxBackendǁgradient__mutmut_1(self, x, *args, **kwargs):
        return jnp.gradient(None, *args, **kwargs)

    def xǁJaxBackendǁgradient__mutmut_2(self, x, *args, **kwargs):
        return jnp.gradient(*args, **kwargs)

    def xǁJaxBackendǁgradient__mutmut_3(self, x, *args, **kwargs):
        return jnp.gradient(x, **kwargs)

    def xǁJaxBackendǁgradient__mutmut_4(self, x, *args, **kwargs):
        return jnp.gradient(x, *args, )
    
    xǁJaxBackendǁgradient__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁgradient__mutmut_1': xǁJaxBackendǁgradient__mutmut_1, 
        'xǁJaxBackendǁgradient__mutmut_2': xǁJaxBackendǁgradient__mutmut_2, 
        'xǁJaxBackendǁgradient__mutmut_3': xǁJaxBackendǁgradient__mutmut_3, 
        'xǁJaxBackendǁgradient__mutmut_4': xǁJaxBackendǁgradient__mutmut_4
    }
    xǁJaxBackendǁgradient__mutmut_orig.__name__ = 'xǁJaxBackendǁgradient'

    def min(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁmin__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁmin__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁmin__mutmut_orig(self, x):
        return jnp.min(x)

    def xǁJaxBackendǁmin__mutmut_1(self, x):
        return jnp.min(None)
    
    xǁJaxBackendǁmin__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁmin__mutmut_1': xǁJaxBackendǁmin__mutmut_1
    }
    xǁJaxBackendǁmin__mutmut_orig.__name__ = 'xǁJaxBackendǁmin'

    def copy(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁcopy__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁcopy__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁcopy__mutmut_orig(self, x):
        return jnp.copy(x)

    def xǁJaxBackendǁcopy__mutmut_1(self, x):
        return jnp.copy(None)
    
    xǁJaxBackendǁcopy__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁcopy__mutmut_1': xǁJaxBackendǁcopy__mutmut_1
    }
    xǁJaxBackendǁcopy__mutmut_orig.__name__ = 'xǁJaxBackendǁcopy'

    def diff(self, a, n=1, axis=-1):
        args = [a, n, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁdiff__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁdiff__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁdiff__mutmut_orig(self, a, n=1, axis=-1):
        return jnp.diff(a, n=n, axis=axis)

    def xǁJaxBackendǁdiff__mutmut_1(self, a, n=2, axis=-1):
        return jnp.diff(a, n=n, axis=axis)

    def xǁJaxBackendǁdiff__mutmut_2(self, a, n=1, axis=-1):
        return jnp.diff(None, n=n, axis=axis)

    def xǁJaxBackendǁdiff__mutmut_3(self, a, n=1, axis=-1):
        return jnp.diff(a, n=None, axis=axis)

    def xǁJaxBackendǁdiff__mutmut_4(self, a, n=1, axis=-1):
        return jnp.diff(a, n=n, axis=None)

    def xǁJaxBackendǁdiff__mutmut_5(self, a, n=1, axis=-1):
        return jnp.diff(n=n, axis=axis)

    def xǁJaxBackendǁdiff__mutmut_6(self, a, n=1, axis=-1):
        return jnp.diff(a, axis=axis)

    def xǁJaxBackendǁdiff__mutmut_7(self, a, n=1, axis=-1):
        return jnp.diff(a, n=n, )
    
    xǁJaxBackendǁdiff__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁdiff__mutmut_1': xǁJaxBackendǁdiff__mutmut_1, 
        'xǁJaxBackendǁdiff__mutmut_2': xǁJaxBackendǁdiff__mutmut_2, 
        'xǁJaxBackendǁdiff__mutmut_3': xǁJaxBackendǁdiff__mutmut_3, 
        'xǁJaxBackendǁdiff__mutmut_4': xǁJaxBackendǁdiff__mutmut_4, 
        'xǁJaxBackendǁdiff__mutmut_5': xǁJaxBackendǁdiff__mutmut_5, 
        'xǁJaxBackendǁdiff__mutmut_6': xǁJaxBackendǁdiff__mutmut_6, 
        'xǁJaxBackendǁdiff__mutmut_7': xǁJaxBackendǁdiff__mutmut_7
    }
    xǁJaxBackendǁdiff__mutmut_orig.__name__ = 'xǁJaxBackendǁdiff'

    def sqrt(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁsqrt__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁsqrt__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁsqrt__mutmut_orig(self, x):
        return jnp.sqrt(x)

    def xǁJaxBackendǁsqrt__mutmut_1(self, x):
        return jnp.sqrt(None)
    
    xǁJaxBackendǁsqrt__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁsqrt__mutmut_1': xǁJaxBackendǁsqrt__mutmut_1
    }
    xǁJaxBackendǁsqrt__mutmut_orig.__name__ = 'xǁJaxBackendǁsqrt'

    def vstack(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁvstack__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁvstack__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁvstack__mutmut_orig(self, x):
        return jnp.vstack(x)

    def xǁJaxBackendǁvstack__mutmut_1(self, x):
        return jnp.vstack(None)
    
    xǁJaxBackendǁvstack__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁvstack__mutmut_1': xǁJaxBackendǁvstack__mutmut_1
    }
    xǁJaxBackendǁvstack__mutmut_orig.__name__ = 'xǁJaxBackendǁvstack'

    def logsumexp(self, x, axis=None):
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁlogsumexp__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁlogsumexp__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁlogsumexp__mutmut_orig(self, x, axis=None):
        return jax.scipy.special.logsumexp(x, axis=axis)

    def xǁJaxBackendǁlogsumexp__mutmut_1(self, x, axis=None):
        return jax.scipy.special.logsumexp(None, axis=axis)

    def xǁJaxBackendǁlogsumexp__mutmut_2(self, x, axis=None):
        return jax.scipy.special.logsumexp(x, axis=None)

    def xǁJaxBackendǁlogsumexp__mutmut_3(self, x, axis=None):
        return jax.scipy.special.logsumexp(axis=axis)

    def xǁJaxBackendǁlogsumexp__mutmut_4(self, x, axis=None):
        return jax.scipy.special.logsumexp(x, )
    
    xǁJaxBackendǁlogsumexp__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁlogsumexp__mutmut_1': xǁJaxBackendǁlogsumexp__mutmut_1, 
        'xǁJaxBackendǁlogsumexp__mutmut_2': xǁJaxBackendǁlogsumexp__mutmut_2, 
        'xǁJaxBackendǁlogsumexp__mutmut_3': xǁJaxBackendǁlogsumexp__mutmut_3, 
        'xǁJaxBackendǁlogsumexp__mutmut_4': xǁJaxBackendǁlogsumexp__mutmut_4
    }
    xǁJaxBackendǁlogsumexp__mutmut_orig.__name__ = 'xǁJaxBackendǁlogsumexp'

    def polyfit(self, x, y, deg):
        args = [x, y, deg]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁpolyfit__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁpolyfit__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁpolyfit__mutmut_orig(self, x, y, deg):
        return jnp.polyfit(x, y, deg)

    def xǁJaxBackendǁpolyfit__mutmut_1(self, x, y, deg):
        return jnp.polyfit(None, y, deg)

    def xǁJaxBackendǁpolyfit__mutmut_2(self, x, y, deg):
        return jnp.polyfit(x, None, deg)

    def xǁJaxBackendǁpolyfit__mutmut_3(self, x, y, deg):
        return jnp.polyfit(x, y, None)

    def xǁJaxBackendǁpolyfit__mutmut_4(self, x, y, deg):
        return jnp.polyfit(y, deg)

    def xǁJaxBackendǁpolyfit__mutmut_5(self, x, y, deg):
        return jnp.polyfit(x, deg)

    def xǁJaxBackendǁpolyfit__mutmut_6(self, x, y, deg):
        return jnp.polyfit(x, y, )
    
    xǁJaxBackendǁpolyfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁpolyfit__mutmut_1': xǁJaxBackendǁpolyfit__mutmut_1, 
        'xǁJaxBackendǁpolyfit__mutmut_2': xǁJaxBackendǁpolyfit__mutmut_2, 
        'xǁJaxBackendǁpolyfit__mutmut_3': xǁJaxBackendǁpolyfit__mutmut_3, 
        'xǁJaxBackendǁpolyfit__mutmut_4': xǁJaxBackendǁpolyfit__mutmut_4, 
        'xǁJaxBackendǁpolyfit__mutmut_5': xǁJaxBackendǁpolyfit__mutmut_5, 
        'xǁJaxBackendǁpolyfit__mutmut_6': xǁJaxBackendǁpolyfit__mutmut_6
    }
    xǁJaxBackendǁpolyfit__mutmut_orig.__name__ = 'xǁJaxBackendǁpolyfit'

    def lstsq(self, x, y, rcond):
        args = [x, y, rcond]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁlstsq__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁlstsq__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁlstsq__mutmut_orig(self, x, y, rcond):
        return jnp.linalg.lstsq(x, y, rcond=rcond)

    def xǁJaxBackendǁlstsq__mutmut_1(self, x, y, rcond):
        return jnp.linalg.lstsq(None, y, rcond=rcond)

    def xǁJaxBackendǁlstsq__mutmut_2(self, x, y, rcond):
        return jnp.linalg.lstsq(x, None, rcond=rcond)

    def xǁJaxBackendǁlstsq__mutmut_3(self, x, y, rcond):
        return jnp.linalg.lstsq(x, y, rcond=None)

    def xǁJaxBackendǁlstsq__mutmut_4(self, x, y, rcond):
        return jnp.linalg.lstsq(y, rcond=rcond)

    def xǁJaxBackendǁlstsq__mutmut_5(self, x, y, rcond):
        return jnp.linalg.lstsq(x, rcond=rcond)

    def xǁJaxBackendǁlstsq__mutmut_6(self, x, y, rcond):
        return jnp.linalg.lstsq(x, y, )
    
    xǁJaxBackendǁlstsq__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁlstsq__mutmut_1': xǁJaxBackendǁlstsq__mutmut_1, 
        'xǁJaxBackendǁlstsq__mutmut_2': xǁJaxBackendǁlstsq__mutmut_2, 
        'xǁJaxBackendǁlstsq__mutmut_3': xǁJaxBackendǁlstsq__mutmut_3, 
        'xǁJaxBackendǁlstsq__mutmut_4': xǁJaxBackendǁlstsq__mutmut_4, 
        'xǁJaxBackendǁlstsq__mutmut_5': xǁJaxBackendǁlstsq__mutmut_5, 
        'xǁJaxBackendǁlstsq__mutmut_6': xǁJaxBackendǁlstsq__mutmut_6
    }
    xǁJaxBackendǁlstsq__mutmut_orig.__name__ = 'xǁJaxBackendǁlstsq'

    def nanmean(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁnanmean__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁnanmean__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁnanmean__mutmut_orig(self, x):
        return jnp.nanmean(x)

    def xǁJaxBackendǁnanmean__mutmut_1(self, x):
        return jnp.nanmean(None)
    
    xǁJaxBackendǁnanmean__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁnanmean__mutmut_1': xǁJaxBackendǁnanmean__mutmut_1
    }
    xǁJaxBackendǁnanmean__mutmut_orig.__name__ = 'xǁJaxBackendǁnanmean'

    def isfinite(self, x):
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁJaxBackendǁisfinite__mutmut_orig'), object.__getattribute__(self, 'xǁJaxBackendǁisfinite__mutmut_mutants'), args, kwargs, self)

    def xǁJaxBackendǁisfinite__mutmut_orig(self, x):
        return jnp.isfinite(x)

    def xǁJaxBackendǁisfinite__mutmut_1(self, x):
        return jnp.isfinite(None)
    
    xǁJaxBackendǁisfinite__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁJaxBackendǁisfinite__mutmut_1': xǁJaxBackendǁisfinite__mutmut_1
    }
    xǁJaxBackendǁisfinite__mutmut_orig.__name__ = 'xǁJaxBackendǁisfinite'
