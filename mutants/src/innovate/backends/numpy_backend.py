from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import odeint
from scipy.special import logsumexp

if TYPE_CHECKING:
    import numpy.typing as npt

    ArrayLike = npt.ArrayLike
    NDArray = npt.NDArray
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


class NumPyBackend:
    def array(self, data: Any) -> np.ndarray:
        args = [data]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁarray__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁarray__mutmut_mutants'), args, kwargs, self)
    def xǁNumPyBackendǁarray__mutmut_orig(self, data: Any) -> np.ndarray:
        """Convert input data to a NumPy array.

        Args:
            data: Input data of any type convertable to array.

        Returns
        -------
            A NumPy array representation of the input data.
        """
        return np.asarray(data)
    def xǁNumPyBackendǁarray__mutmut_1(self, data: Any) -> np.ndarray:
        """Convert input data to a NumPy array.

        Args:
            data: Input data of any type convertable to array.

        Returns
        -------
            A NumPy array representation of the input data.
        """
        return np.asarray(None)
    
    xǁNumPyBackendǁarray__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁarray__mutmut_1': xǁNumPyBackendǁarray__mutmut_1
    }
    xǁNumPyBackendǁarray__mutmut_orig.__name__ = 'xǁNumPyBackendǁarray'

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
        args = [a, axis, dtype, out, keepdims, initial, where]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁsum__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁsum__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁsum__mutmut_orig(
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

    def xǁNumPyBackendǁsum__mutmut_1(
        self,
        a: np.ndarray | Sequence,
        axis: int | tuple | None = None,
        dtype: type | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = True,
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

    def xǁNumPyBackendǁsum__mutmut_2(
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
        kwargs = None
        if initial is not None:
            kwargs["initial"] = initial
        if where is not None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_3(
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
            "XXaxisXX": axis,
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

    def xǁNumPyBackendǁsum__mutmut_4(
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
            "AXIS": axis,
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

    def xǁNumPyBackendǁsum__mutmut_5(
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
            "XXdtypeXX": dtype,
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

    def xǁNumPyBackendǁsum__mutmut_6(
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
            "DTYPE": dtype,
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

    def xǁNumPyBackendǁsum__mutmut_7(
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
            "XXoutXX": out,
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

    def xǁNumPyBackendǁsum__mutmut_8(
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
            "OUT": out,
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

    def xǁNumPyBackendǁsum__mutmut_9(
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
            "XXkeepdimsXX": keepdims,
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

    def xǁNumPyBackendǁsum__mutmut_10(
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
            "KEEPDIMS": keepdims,
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

    def xǁNumPyBackendǁsum__mutmut_11(
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
        if initial is None:
            kwargs["initial"] = initial
        if where is not None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_12(
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
            kwargs["initial"] = None
        if where is not None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_13(
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
            kwargs["XXinitialXX"] = initial
        if where is not None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_14(
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
            kwargs["INITIAL"] = initial
        if where is not None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_15(
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
        if where is None:
            kwargs["where"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_16(
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
            kwargs["where"] = None
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_17(
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
            kwargs["XXwhereXX"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_18(
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
            kwargs["WHERE"] = where
        result = np.sum(a, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_19(
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
        result = None

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_20(
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
        result = np.sum(None, **kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_21(
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
        result = np.sum(**kwargs)

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_22(
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
        result = np.sum(a, )

        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_23(
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
        if axis is None or not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_24(
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
        if axis is not None and not keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_25(
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
        if axis is None and keepdims:
            return float(result)
        return result

    def xǁNumPyBackendǁsum__mutmut_26(
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
            return float(None)
        return result
    
    xǁNumPyBackendǁsum__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁsum__mutmut_1': xǁNumPyBackendǁsum__mutmut_1, 
        'xǁNumPyBackendǁsum__mutmut_2': xǁNumPyBackendǁsum__mutmut_2, 
        'xǁNumPyBackendǁsum__mutmut_3': xǁNumPyBackendǁsum__mutmut_3, 
        'xǁNumPyBackendǁsum__mutmut_4': xǁNumPyBackendǁsum__mutmut_4, 
        'xǁNumPyBackendǁsum__mutmut_5': xǁNumPyBackendǁsum__mutmut_5, 
        'xǁNumPyBackendǁsum__mutmut_6': xǁNumPyBackendǁsum__mutmut_6, 
        'xǁNumPyBackendǁsum__mutmut_7': xǁNumPyBackendǁsum__mutmut_7, 
        'xǁNumPyBackendǁsum__mutmut_8': xǁNumPyBackendǁsum__mutmut_8, 
        'xǁNumPyBackendǁsum__mutmut_9': xǁNumPyBackendǁsum__mutmut_9, 
        'xǁNumPyBackendǁsum__mutmut_10': xǁNumPyBackendǁsum__mutmut_10, 
        'xǁNumPyBackendǁsum__mutmut_11': xǁNumPyBackendǁsum__mutmut_11, 
        'xǁNumPyBackendǁsum__mutmut_12': xǁNumPyBackendǁsum__mutmut_12, 
        'xǁNumPyBackendǁsum__mutmut_13': xǁNumPyBackendǁsum__mutmut_13, 
        'xǁNumPyBackendǁsum__mutmut_14': xǁNumPyBackendǁsum__mutmut_14, 
        'xǁNumPyBackendǁsum__mutmut_15': xǁNumPyBackendǁsum__mutmut_15, 
        'xǁNumPyBackendǁsum__mutmut_16': xǁNumPyBackendǁsum__mutmut_16, 
        'xǁNumPyBackendǁsum__mutmut_17': xǁNumPyBackendǁsum__mutmut_17, 
        'xǁNumPyBackendǁsum__mutmut_18': xǁNumPyBackendǁsum__mutmut_18, 
        'xǁNumPyBackendǁsum__mutmut_19': xǁNumPyBackendǁsum__mutmut_19, 
        'xǁNumPyBackendǁsum__mutmut_20': xǁNumPyBackendǁsum__mutmut_20, 
        'xǁNumPyBackendǁsum__mutmut_21': xǁNumPyBackendǁsum__mutmut_21, 
        'xǁNumPyBackendǁsum__mutmut_22': xǁNumPyBackendǁsum__mutmut_22, 
        'xǁNumPyBackendǁsum__mutmut_23': xǁNumPyBackendǁsum__mutmut_23, 
        'xǁNumPyBackendǁsum__mutmut_24': xǁNumPyBackendǁsum__mutmut_24, 
        'xǁNumPyBackendǁsum__mutmut_25': xǁNumPyBackendǁsum__mutmut_25, 
        'xǁNumPyBackendǁsum__mutmut_26': xǁNumPyBackendǁsum__mutmut_26
    }
    xǁNumPyBackendǁsum__mutmut_orig.__name__ = 'xǁNumPyBackendǁsum'

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
        args = [a, axis, dtype, out, keepdims]# type: ignore
        kwargs = {'where': where}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁmean__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁmean__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁmean__mutmut_orig(
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

    def xǁNumPyBackendǁmean__mutmut_1(
        self,
        a: np.ndarray | Sequence,
        axis: int | tuple | None = None,
        dtype: type | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = True,
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

    def xǁNumPyBackendǁmean__mutmut_2(
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
        kwargs = None
        if where is not None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_3(
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
            "XXaxisXX": axis,
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

    def xǁNumPyBackendǁmean__mutmut_4(
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
            "AXIS": axis,
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

    def xǁNumPyBackendǁmean__mutmut_5(
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
            "XXdtypeXX": dtype,
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

    def xǁNumPyBackendǁmean__mutmut_6(
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
            "DTYPE": dtype,
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

    def xǁNumPyBackendǁmean__mutmut_7(
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
            "XXoutXX": out,
            "keepdims": keepdims,
        }
        if where is not None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_8(
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
            "OUT": out,
            "keepdims": keepdims,
        }
        if where is not None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_9(
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
            "XXkeepdimsXX": keepdims,
        }
        if where is not None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_10(
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
            "KEEPDIMS": keepdims,
        }
        if where is not None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_11(
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
        if where is None:
            kwargs["where"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_12(
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
            kwargs["where"] = None
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_13(
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
            kwargs["XXwhereXX"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_14(
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
            kwargs["WHERE"] = where
        result = np.mean(a, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_15(
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
        result = None
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_16(
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
        result = np.mean(None, **kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_17(
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
        result = np.mean(**kwargs)
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_18(
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
        result = np.mean(a, )
        # Return float if scalar (when axis is None), otherwise return array
        if axis is None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_19(
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
        if axis is None or not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_20(
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
        if axis is not None and not keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_21(
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
        if axis is None and keepdims:
            return float(result)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_22(
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
            return float(None)
        return np.asarray(result)

    def xǁNumPyBackendǁmean__mutmut_23(
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
        return np.asarray(None)
    
    xǁNumPyBackendǁmean__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁmean__mutmut_1': xǁNumPyBackendǁmean__mutmut_1, 
        'xǁNumPyBackendǁmean__mutmut_2': xǁNumPyBackendǁmean__mutmut_2, 
        'xǁNumPyBackendǁmean__mutmut_3': xǁNumPyBackendǁmean__mutmut_3, 
        'xǁNumPyBackendǁmean__mutmut_4': xǁNumPyBackendǁmean__mutmut_4, 
        'xǁNumPyBackendǁmean__mutmut_5': xǁNumPyBackendǁmean__mutmut_5, 
        'xǁNumPyBackendǁmean__mutmut_6': xǁNumPyBackendǁmean__mutmut_6, 
        'xǁNumPyBackendǁmean__mutmut_7': xǁNumPyBackendǁmean__mutmut_7, 
        'xǁNumPyBackendǁmean__mutmut_8': xǁNumPyBackendǁmean__mutmut_8, 
        'xǁNumPyBackendǁmean__mutmut_9': xǁNumPyBackendǁmean__mutmut_9, 
        'xǁNumPyBackendǁmean__mutmut_10': xǁNumPyBackendǁmean__mutmut_10, 
        'xǁNumPyBackendǁmean__mutmut_11': xǁNumPyBackendǁmean__mutmut_11, 
        'xǁNumPyBackendǁmean__mutmut_12': xǁNumPyBackendǁmean__mutmut_12, 
        'xǁNumPyBackendǁmean__mutmut_13': xǁNumPyBackendǁmean__mutmut_13, 
        'xǁNumPyBackendǁmean__mutmut_14': xǁNumPyBackendǁmean__mutmut_14, 
        'xǁNumPyBackendǁmean__mutmut_15': xǁNumPyBackendǁmean__mutmut_15, 
        'xǁNumPyBackendǁmean__mutmut_16': xǁNumPyBackendǁmean__mutmut_16, 
        'xǁNumPyBackendǁmean__mutmut_17': xǁNumPyBackendǁmean__mutmut_17, 
        'xǁNumPyBackendǁmean__mutmut_18': xǁNumPyBackendǁmean__mutmut_18, 
        'xǁNumPyBackendǁmean__mutmut_19': xǁNumPyBackendǁmean__mutmut_19, 
        'xǁNumPyBackendǁmean__mutmut_20': xǁNumPyBackendǁmean__mutmut_20, 
        'xǁNumPyBackendǁmean__mutmut_21': xǁNumPyBackendǁmean__mutmut_21, 
        'xǁNumPyBackendǁmean__mutmut_22': xǁNumPyBackendǁmean__mutmut_22, 
        'xǁNumPyBackendǁmean__mutmut_23': xǁNumPyBackendǁmean__mutmut_23
    }
    xǁNumPyBackendǁmean__mutmut_orig.__name__ = 'xǁNumPyBackendǁmean'

    def where(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
        args = [condition, x, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁwhere__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁwhere__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁwhere__mutmut_orig(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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

    def xǁNumPyBackendǁwhere__mutmut_1(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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
        return np.where(None, x, y)

    def xǁNumPyBackendǁwhere__mutmut_2(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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
        return np.where(condition, None, y)

    def xǁNumPyBackendǁwhere__mutmut_3(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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
        return np.where(condition, x, None)

    def xǁNumPyBackendǁwhere__mutmut_4(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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
        return np.where(x, y)

    def xǁNumPyBackendǁwhere__mutmut_5(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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
        return np.where(condition, y)

    def xǁNumPyBackendǁwhere__mutmut_6(self, condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
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
        return np.where(condition, x, )
    
    xǁNumPyBackendǁwhere__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁwhere__mutmut_1': xǁNumPyBackendǁwhere__mutmut_1, 
        'xǁNumPyBackendǁwhere__mutmut_2': xǁNumPyBackendǁwhere__mutmut_2, 
        'xǁNumPyBackendǁwhere__mutmut_3': xǁNumPyBackendǁwhere__mutmut_3, 
        'xǁNumPyBackendǁwhere__mutmut_4': xǁNumPyBackendǁwhere__mutmut_4, 
        'xǁNumPyBackendǁwhere__mutmut_5': xǁNumPyBackendǁwhere__mutmut_5, 
        'xǁNumPyBackendǁwhere__mutmut_6': xǁNumPyBackendǁwhere__mutmut_6
    }
    xǁNumPyBackendǁwhere__mutmut_orig.__name__ = 'xǁNumPyBackendǁwhere'

    def diff(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        args = [a, n, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁdiff__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁdiff__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁdiff__mutmut_orig(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=n, axis=axis)

    def xǁNumPyBackendǁdiff__mutmut_1(self, a: np.ndarray, n: int = 2, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=n, axis=axis)

    def xǁNumPyBackendǁdiff__mutmut_2(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(None, n=n, axis=axis)

    def xǁNumPyBackendǁdiff__mutmut_3(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=None, axis=axis)

    def xǁNumPyBackendǁdiff__mutmut_4(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=n, axis=None)

    def xǁNumPyBackendǁdiff__mutmut_5(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(n=n, axis=axis)

    def xǁNumPyBackendǁdiff__mutmut_6(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, axis=axis)

    def xǁNumPyBackendǁdiff__mutmut_7(self, a: np.ndarray, n: int = 1, axis: int = -1) -> np.ndarray:
        return np.diff(a, n=n, )
    
    xǁNumPyBackendǁdiff__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁdiff__mutmut_1': xǁNumPyBackendǁdiff__mutmut_1, 
        'xǁNumPyBackendǁdiff__mutmut_2': xǁNumPyBackendǁdiff__mutmut_2, 
        'xǁNumPyBackendǁdiff__mutmut_3': xǁNumPyBackendǁdiff__mutmut_3, 
        'xǁNumPyBackendǁdiff__mutmut_4': xǁNumPyBackendǁdiff__mutmut_4, 
        'xǁNumPyBackendǁdiff__mutmut_5': xǁNumPyBackendǁdiff__mutmut_5, 
        'xǁNumPyBackendǁdiff__mutmut_6': xǁNumPyBackendǁdiff__mutmut_6, 
        'xǁNumPyBackendǁdiff__mutmut_7': xǁNumPyBackendǁdiff__mutmut_7
    }
    xǁNumPyBackendǁdiff__mutmut_orig.__name__ = 'xǁNumPyBackendǁdiff'

    def log(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁlog__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁlog__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁlog__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def xǁNumPyBackendǁlog__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.log(None)
    
    xǁNumPyBackendǁlog__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁlog__mutmut_1': xǁNumPyBackendǁlog__mutmut_1
    }
    xǁNumPyBackendǁlog__mutmut_orig.__name__ = 'xǁNumPyBackendǁlog'

    def logsumexp(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁlogsumexp__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁlogsumexp__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁlogsumexp__mutmut_orig(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return logsumexp(x, axis=axis)

    def xǁNumPyBackendǁlogsumexp__mutmut_1(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return logsumexp(None, axis=axis)

    def xǁNumPyBackendǁlogsumexp__mutmut_2(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return logsumexp(x, axis=None)

    def xǁNumPyBackendǁlogsumexp__mutmut_3(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return logsumexp(axis=axis)

    def xǁNumPyBackendǁlogsumexp__mutmut_4(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        return logsumexp(x, )
    
    xǁNumPyBackendǁlogsumexp__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁlogsumexp__mutmut_1': xǁNumPyBackendǁlogsumexp__mutmut_1, 
        'xǁNumPyBackendǁlogsumexp__mutmut_2': xǁNumPyBackendǁlogsumexp__mutmut_2, 
        'xǁNumPyBackendǁlogsumexp__mutmut_3': xǁNumPyBackendǁlogsumexp__mutmut_3, 
        'xǁNumPyBackendǁlogsumexp__mutmut_4': xǁNumPyBackendǁlogsumexp__mutmut_4
    }
    xǁNumPyBackendǁlogsumexp__mutmut_orig.__name__ = 'xǁNumPyBackendǁlogsumexp'

    def solve_ode(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        args = [f, y0, t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁsolve_ode__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁsolve_ode__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁsolve_ode__mutmut_orig(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(f, y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_1(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = None
        t_array = np.asarray(t)
        sol = odeint(f, y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_2(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(None)
        t_array = np.asarray(t)
        sol = odeint(f, y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_3(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = None
        sol = odeint(f, y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_4(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(None)
        sol = odeint(f, y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_5(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = None
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_6(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(None, y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_7(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(f, None, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_8(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(f, y0_array, None)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_9(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(y0_array, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_10(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(f, t_array)
        return sol

    def xǁNumPyBackendǁsolve_ode__mutmut_11(self, f: Any, y0: Sequence | np.ndarray, t: Sequence | np.ndarray) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        y0_array = np.asarray(y0)
        t_array = np.asarray(t)
        sol = odeint(f, y0_array, )
        return sol
    
    xǁNumPyBackendǁsolve_ode__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁsolve_ode__mutmut_1': xǁNumPyBackendǁsolve_ode__mutmut_1, 
        'xǁNumPyBackendǁsolve_ode__mutmut_2': xǁNumPyBackendǁsolve_ode__mutmut_2, 
        'xǁNumPyBackendǁsolve_ode__mutmut_3': xǁNumPyBackendǁsolve_ode__mutmut_3, 
        'xǁNumPyBackendǁsolve_ode__mutmut_4': xǁNumPyBackendǁsolve_ode__mutmut_4, 
        'xǁNumPyBackendǁsolve_ode__mutmut_5': xǁNumPyBackendǁsolve_ode__mutmut_5, 
        'xǁNumPyBackendǁsolve_ode__mutmut_6': xǁNumPyBackendǁsolve_ode__mutmut_6, 
        'xǁNumPyBackendǁsolve_ode__mutmut_7': xǁNumPyBackendǁsolve_ode__mutmut_7, 
        'xǁNumPyBackendǁsolve_ode__mutmut_8': xǁNumPyBackendǁsolve_ode__mutmut_8, 
        'xǁNumPyBackendǁsolve_ode__mutmut_9': xǁNumPyBackendǁsolve_ode__mutmut_9, 
        'xǁNumPyBackendǁsolve_ode__mutmut_10': xǁNumPyBackendǁsolve_ode__mutmut_10, 
        'xǁNumPyBackendǁsolve_ode__mutmut_11': xǁNumPyBackendǁsolve_ode__mutmut_11
    }
    xǁNumPyBackendǁsolve_ode__mutmut_orig.__name__ = 'xǁNumPyBackendǁsolve_ode'

    def stack(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        args = [arrays]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁstack__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁstack__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁstack__mutmut_orig(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(arrays)

    def xǁNumPyBackendǁstack__mutmut_1(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(None)
    
    xǁNumPyBackendǁstack__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁstack__mutmut_1': xǁNumPyBackendǁstack__mutmut_1
    }
    xǁNumPyBackendǁstack__mutmut_orig.__name__ = 'xǁNumPyBackendǁstack'

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        args = [a, b]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁmatmul__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁmatmul__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁmatmul__mutmut_orig(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def xǁNumPyBackendǁmatmul__mutmut_1(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(None, b)

    def xǁNumPyBackendǁmatmul__mutmut_2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, None)

    def xǁNumPyBackendǁmatmul__mutmut_3(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(b)

    def xǁNumPyBackendǁmatmul__mutmut_4(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, )
    
    xǁNumPyBackendǁmatmul__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁmatmul__mutmut_1': xǁNumPyBackendǁmatmul__mutmut_1, 
        'xǁNumPyBackendǁmatmul__mutmut_2': xǁNumPyBackendǁmatmul__mutmut_2, 
        'xǁNumPyBackendǁmatmul__mutmut_3': xǁNumPyBackendǁmatmul__mutmut_3, 
        'xǁNumPyBackendǁmatmul__mutmut_4': xǁNumPyBackendǁmatmul__mutmut_4
    }
    xǁNumPyBackendǁmatmul__mutmut_orig.__name__ = 'xǁNumPyBackendǁmatmul'

    def zeros(self, shape: int | Sequence[int]) -> np.ndarray:
        args = [shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁzeros__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁzeros__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁzeros__mutmut_orig(self, shape: int | Sequence[int]) -> np.ndarray:
        """Return a new array of given shape and type, filled with zeros.

        Args:
            shape: Shape of the new array, e.g., (2, 3) or 2.

        Returns
        -------
            Array of zeros with the specified shape.
        """
        return np.zeros(shape)

    def xǁNumPyBackendǁzeros__mutmut_1(self, shape: int | Sequence[int]) -> np.ndarray:
        """Return a new array of given shape and type, filled with zeros.

        Args:
            shape: Shape of the new array, e.g., (2, 3) or 2.

        Returns
        -------
            Array of zeros with the specified shape.
        """
        return np.zeros(None)
    
    xǁNumPyBackendǁzeros__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁzeros__mutmut_1': xǁNumPyBackendǁzeros__mutmut_1
    }
    xǁNumPyBackendǁzeros__mutmut_orig.__name__ = 'xǁNumPyBackendǁzeros'

    def ones(self, shape: int | Sequence[int]) -> np.ndarray:
        args = [shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁones__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁones__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁones__mutmut_orig(self, shape: int | Sequence[int]) -> np.ndarray:
        """Return a new array of given shape and type, filled with ones.

        Args:
            shape: Shape of the new array, e.g., (2, 3) or 2.

        Returns
        -------
            Array of ones with the specified shape.
        """
        return np.ones(shape)

    def xǁNumPyBackendǁones__mutmut_1(self, shape: int | Sequence[int]) -> np.ndarray:
        """Return a new array of given shape and type, filled with ones.

        Args:
            shape: Shape of the new array, e.g., (2, 3) or 2.

        Returns
        -------
            Array of ones with the specified shape.
        """
        return np.ones(None)
    
    xǁNumPyBackendǁones__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁones__mutmut_1': xǁNumPyBackendǁones__mutmut_1
    }
    xǁNumPyBackendǁones__mutmut_orig.__name__ = 'xǁNumPyBackendǁones'

    def max(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁmax__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁmax__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁmax__mutmut_orig(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return np.max(x, axis=axis)

    def xǁNumPyBackendǁmax__mutmut_1(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return np.max(None, axis=axis)

    def xǁNumPyBackendǁmax__mutmut_2(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return np.max(x, axis=None)

    def xǁNumPyBackendǁmax__mutmut_3(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return np.max(axis=axis)

    def xǁNumPyBackendǁmax__mutmut_4(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the maximum of an array or maximum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Maximum of the array elements, float if scalar, array if axis specified.
        """
        return np.max(x, )
    
    xǁNumPyBackendǁmax__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁmax__mutmut_1': xǁNumPyBackendǁmax__mutmut_1, 
        'xǁNumPyBackendǁmax__mutmut_2': xǁNumPyBackendǁmax__mutmut_2, 
        'xǁNumPyBackendǁmax__mutmut_3': xǁNumPyBackendǁmax__mutmut_3, 
        'xǁNumPyBackendǁmax__mutmut_4': xǁNumPyBackendǁmax__mutmut_4
    }
    xǁNumPyBackendǁmax__mutmut_orig.__name__ = 'xǁNumPyBackendǁmax'

    def median(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁmedian__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁmedian__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁmedian__mutmut_orig(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
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

    def xǁNumPyBackendǁmedian__mutmut_1(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array or sequence containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return np.median(None, axis=axis)

    def xǁNumPyBackendǁmedian__mutmut_2(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array or sequence containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return np.median(x, axis=None)

    def xǁNumPyBackendǁmedian__mutmut_3(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array or sequence containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return np.median(axis=axis)

    def xǁNumPyBackendǁmedian__mutmut_4(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Compute the median along the specified axis.

        Args:
            x: Input array or sequence containing numbers.
            axis: Axis or axes along which the medians are computed.
                  None (default) computes the median of the flattened array.

        Returns
        -------
            Median of the array elements, float if scalar, array if axis specified.
        """
        return np.median(x, )
    
    xǁNumPyBackendǁmedian__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁmedian__mutmut_1': xǁNumPyBackendǁmedian__mutmut_1, 
        'xǁNumPyBackendǁmedian__mutmut_2': xǁNumPyBackendǁmedian__mutmut_2, 
        'xǁNumPyBackendǁmedian__mutmut_3': xǁNumPyBackendǁmedian__mutmut_3, 
        'xǁNumPyBackendǁmedian__mutmut_4': xǁNumPyBackendǁmedian__mutmut_4
    }
    xǁNumPyBackendǁmedian__mutmut_orig.__name__ = 'xǁNumPyBackendǁmedian'

    def interp(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        args = [x, xp, fp]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁinterp__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁinterp__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁinterp__mutmut_orig(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(x, xp, fp)

    def xǁNumPyBackendǁinterp__mutmut_1(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(None, xp, fp)

    def xǁNumPyBackendǁinterp__mutmut_2(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(x, None, fp)

    def xǁNumPyBackendǁinterp__mutmut_3(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(x, xp, None)

    def xǁNumPyBackendǁinterp__mutmut_4(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(xp, fp)

    def xǁNumPyBackendǁinterp__mutmut_5(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(x, fp)

    def xǁNumPyBackendǁinterp__mutmut_6(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        return np.interp(x, xp, )
    
    xǁNumPyBackendǁinterp__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁinterp__mutmut_1': xǁNumPyBackendǁinterp__mutmut_1, 
        'xǁNumPyBackendǁinterp__mutmut_2': xǁNumPyBackendǁinterp__mutmut_2, 
        'xǁNumPyBackendǁinterp__mutmut_3': xǁNumPyBackendǁinterp__mutmut_3, 
        'xǁNumPyBackendǁinterp__mutmut_4': xǁNumPyBackendǁinterp__mutmut_4, 
        'xǁNumPyBackendǁinterp__mutmut_5': xǁNumPyBackendǁinterp__mutmut_5, 
        'xǁNumPyBackendǁinterp__mutmut_6': xǁNumPyBackendǁinterp__mutmut_6
    }
    xǁNumPyBackendǁinterp__mutmut_orig.__name__ = 'xǁNumPyBackendǁinterp'

    def jit(self, f: Any) -> Any:
        return f

    def vmap(self, f: Any) -> Any:
        args = [f]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁvmap__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁvmap__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁvmap__mutmut_orig(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(params, t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_1(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array(None)

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_2(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(None, t) for p, t in zip(params, t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_3(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, None) for p, t in zip(params, t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_4(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(t) for p, t in zip(params, t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_5(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, ) for p, t in zip(params, t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_6(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(None, t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_7(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(params, None)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_8(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(t_batched)])

        return mapped_f

    def xǁNumPyBackendǁvmap__mutmut_9(self, f: Any) -> Any:
        def mapped_f(params, t_batched):
            return np.array([f(p, t) for p, t in zip(params, )])

        return mapped_f
    
    xǁNumPyBackendǁvmap__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁvmap__mutmut_1': xǁNumPyBackendǁvmap__mutmut_1, 
        'xǁNumPyBackendǁvmap__mutmut_2': xǁNumPyBackendǁvmap__mutmut_2, 
        'xǁNumPyBackendǁvmap__mutmut_3': xǁNumPyBackendǁvmap__mutmut_3, 
        'xǁNumPyBackendǁvmap__mutmut_4': xǁNumPyBackendǁvmap__mutmut_4, 
        'xǁNumPyBackendǁvmap__mutmut_5': xǁNumPyBackendǁvmap__mutmut_5, 
        'xǁNumPyBackendǁvmap__mutmut_6': xǁNumPyBackendǁvmap__mutmut_6, 
        'xǁNumPyBackendǁvmap__mutmut_7': xǁNumPyBackendǁvmap__mutmut_7, 
        'xǁNumPyBackendǁvmap__mutmut_8': xǁNumPyBackendǁvmap__mutmut_8, 
        'xǁNumPyBackendǁvmap__mutmut_9': xǁNumPyBackendǁvmap__mutmut_9
    }
    xǁNumPyBackendǁvmap__mutmut_orig.__name__ = 'xǁNumPyBackendǁvmap'

    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁzeros_like__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁzeros_like__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁzeros_like__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def xǁNumPyBackendǁzeros_like__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(None)
    
    xǁNumPyBackendǁzeros_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁzeros_like__mutmut_1': xǁNumPyBackendǁzeros_like__mutmut_1
    }
    xǁNumPyBackendǁzeros_like__mutmut_orig.__name__ = 'xǁNumPyBackendǁzeros_like'

    def ones_like(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁones_like__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁones_like__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁones_like__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def xǁNumPyBackendǁones_like__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(None)
    
    xǁNumPyBackendǁones_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁones_like__mutmut_1': xǁNumPyBackendǁones_like__mutmut_1
    }
    xǁNumPyBackendǁones_like__mutmut_orig.__name__ = 'xǁNumPyBackendǁones_like'

    def ravel(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁravel__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁravel__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁravel__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.ravel(x)

    def xǁNumPyBackendǁravel__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.ravel(None)
    
    xǁNumPyBackendǁravel__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁravel__mutmut_1': xǁNumPyBackendǁravel__mutmut_1
    }
    xǁNumPyBackendǁravel__mutmut_orig.__name__ = 'xǁNumPyBackendǁravel'

    def argmin(self, x: np.ndarray) -> int:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁargmin__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁargmin__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁargmin__mutmut_orig(self, x: np.ndarray) -> int:
        return int(np.argmin(x))

    def xǁNumPyBackendǁargmin__mutmut_1(self, x: np.ndarray) -> int:
        return int(None)

    def xǁNumPyBackendǁargmin__mutmut_2(self, x: np.ndarray) -> int:
        return int(np.argmin(None))
    
    xǁNumPyBackendǁargmin__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁargmin__mutmut_1': xǁNumPyBackendǁargmin__mutmut_1, 
        'xǁNumPyBackendǁargmin__mutmut_2': xǁNumPyBackendǁargmin__mutmut_2
    }
    xǁNumPyBackendǁargmin__mutmut_orig.__name__ = 'xǁNumPyBackendǁargmin'

    def abs(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁabs__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁabs__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁabs__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def xǁNumPyBackendǁabs__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.abs(None)
    
    xǁNumPyBackendǁabs__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁabs__mutmut_1': xǁNumPyBackendǁabs__mutmut_1
    }
    xǁNumPyBackendǁabs__mutmut_orig.__name__ = 'xǁNumPyBackendǁabs'

    def gradient(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        args = [x, *args]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁgradient__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁgradient__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁgradient__mutmut_orig(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(x, *args, **kwargs)

    def xǁNumPyBackendǁgradient__mutmut_1(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(None, *args, **kwargs)

    def xǁNumPyBackendǁgradient__mutmut_2(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(*args, **kwargs)

    def xǁNumPyBackendǁgradient__mutmut_3(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(x, **kwargs)

    def xǁNumPyBackendǁgradient__mutmut_4(self, x: np.ndarray | Sequence, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.gradient(x, *args, )
    
    xǁNumPyBackendǁgradient__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁgradient__mutmut_1': xǁNumPyBackendǁgradient__mutmut_1, 
        'xǁNumPyBackendǁgradient__mutmut_2': xǁNumPyBackendǁgradient__mutmut_2, 
        'xǁNumPyBackendǁgradient__mutmut_3': xǁNumPyBackendǁgradient__mutmut_3, 
        'xǁNumPyBackendǁgradient__mutmut_4': xǁNumPyBackendǁgradient__mutmut_4
    }
    xǁNumPyBackendǁgradient__mutmut_orig.__name__ = 'xǁNumPyBackendǁgradient'

    def clip(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        args = [x, a_min, a_max]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁclip__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁclip__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁclip__mutmut_orig(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, a_min, a_max)

    def xǁNumPyBackendǁclip__mutmut_1(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(None, a_min, a_max)

    def xǁNumPyBackendǁclip__mutmut_2(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, None, a_max)

    def xǁNumPyBackendǁclip__mutmut_3(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, a_min, None)

    def xǁNumPyBackendǁclip__mutmut_4(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(a_min, a_max)

    def xǁNumPyBackendǁclip__mutmut_5(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, a_max)

    def xǁNumPyBackendǁclip__mutmut_6(self, x: np.ndarray, a_min: Any, a_max: Any) -> np.ndarray:
        return np.clip(x, a_min, )
    
    xǁNumPyBackendǁclip__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁclip__mutmut_1': xǁNumPyBackendǁclip__mutmut_1, 
        'xǁNumPyBackendǁclip__mutmut_2': xǁNumPyBackendǁclip__mutmut_2, 
        'xǁNumPyBackendǁclip__mutmut_3': xǁNumPyBackendǁclip__mutmut_3, 
        'xǁNumPyBackendǁclip__mutmut_4': xǁNumPyBackendǁclip__mutmut_4, 
        'xǁNumPyBackendǁclip__mutmut_5': xǁNumPyBackendǁclip__mutmut_5, 
        'xǁNumPyBackendǁclip__mutmut_6': xǁNumPyBackendǁclip__mutmut_6
    }
    xǁNumPyBackendǁclip__mutmut_orig.__name__ = 'xǁNumPyBackendǁclip'

    def min(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        args = [x, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁmin__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁmin__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁmin__mutmut_orig(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the minimum of an array or minimum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Minimum of the array elements, float if scalar, array if axis specified.
        """
        return np.min(x, axis=axis)

    def xǁNumPyBackendǁmin__mutmut_1(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the minimum of an array or minimum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Minimum of the array elements, float if scalar, array if axis specified.
        """
        return np.min(None, axis=axis)

    def xǁNumPyBackendǁmin__mutmut_2(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the minimum of an array or minimum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Minimum of the array elements, float if scalar, array if axis specified.
        """
        return np.min(x, axis=None)

    def xǁNumPyBackendǁmin__mutmut_3(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the minimum of an array or minimum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Minimum of the array elements, float if scalar, array if axis specified.
        """
        return np.min(axis=axis)

    def xǁNumPyBackendǁmin__mutmut_4(self, x: np.ndarray | Sequence, axis: int | tuple | None = None) -> float | np.ndarray:
        """Return the minimum of an array or minimum along an axis.

        Args:
            x: Input array or sequence.
            axis: Axis along which to operate. If None, the flattened array is used.

        Returns
        -------
            Minimum of the array elements, float if scalar, array if axis specified.
        """
        return np.min(x, )
    
    xǁNumPyBackendǁmin__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁmin__mutmut_1': xǁNumPyBackendǁmin__mutmut_1, 
        'xǁNumPyBackendǁmin__mutmut_2': xǁNumPyBackendǁmin__mutmut_2, 
        'xǁNumPyBackendǁmin__mutmut_3': xǁNumPyBackendǁmin__mutmut_3, 
        'xǁNumPyBackendǁmin__mutmut_4': xǁNumPyBackendǁmin__mutmut_4
    }
    xǁNumPyBackendǁmin__mutmut_orig.__name__ = 'xǁNumPyBackendǁmin'

    def copy(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁcopy__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁcopy__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁcopy__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.copy(x)

    def xǁNumPyBackendǁcopy__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.copy(None)
    
    xǁNumPyBackendǁcopy__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁcopy__mutmut_1': xǁNumPyBackendǁcopy__mutmut_1
    }
    xǁNumPyBackendǁcopy__mutmut_orig.__name__ = 'xǁNumPyBackendǁcopy'

    def vstack(self, x: Sequence[np.ndarray]) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁvstack__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁvstack__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁvstack__mutmut_orig(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return np.vstack(x)

    def xǁNumPyBackendǁvstack__mutmut_1(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return np.vstack(None)
    
    xǁNumPyBackendǁvstack__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁvstack__mutmut_1': xǁNumPyBackendǁvstack__mutmut_1
    }
    xǁNumPyBackendǁvstack__mutmut_orig.__name__ = 'xǁNumPyBackendǁvstack'

    def polyfit(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        args = [x, y, deg]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁpolyfit__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁpolyfit__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁpolyfit__mutmut_orig(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, y, deg)

    def xǁNumPyBackendǁpolyfit__mutmut_1(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(None, y, deg)

    def xǁNumPyBackendǁpolyfit__mutmut_2(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, None, deg)

    def xǁNumPyBackendǁpolyfit__mutmut_3(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, y, None)

    def xǁNumPyBackendǁpolyfit__mutmut_4(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(y, deg)

    def xǁNumPyBackendǁpolyfit__mutmut_5(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, deg)

    def xǁNumPyBackendǁpolyfit__mutmut_6(self, x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
        return np.polyfit(x, y, )
    
    xǁNumPyBackendǁpolyfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁpolyfit__mutmut_1': xǁNumPyBackendǁpolyfit__mutmut_1, 
        'xǁNumPyBackendǁpolyfit__mutmut_2': xǁNumPyBackendǁpolyfit__mutmut_2, 
        'xǁNumPyBackendǁpolyfit__mutmut_3': xǁNumPyBackendǁpolyfit__mutmut_3, 
        'xǁNumPyBackendǁpolyfit__mutmut_4': xǁNumPyBackendǁpolyfit__mutmut_4, 
        'xǁNumPyBackendǁpolyfit__mutmut_5': xǁNumPyBackendǁpolyfit__mutmut_5, 
        'xǁNumPyBackendǁpolyfit__mutmut_6': xǁNumPyBackendǁpolyfit__mutmut_6
    }
    xǁNumPyBackendǁpolyfit__mutmut_orig.__name__ = 'xǁNumPyBackendǁpolyfit'

    def lstsq(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        args = [x, y, rcond]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁlstsq__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁlstsq__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁlstsq__mutmut_orig(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(x, y, rcond=rcond)

    def xǁNumPyBackendǁlstsq__mutmut_1(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(None, y, rcond=rcond)

    def xǁNumPyBackendǁlstsq__mutmut_2(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(x, None, rcond=rcond)

    def xǁNumPyBackendǁlstsq__mutmut_3(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(x, y, rcond=None)

    def xǁNumPyBackendǁlstsq__mutmut_4(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(y, rcond=rcond)

    def xǁNumPyBackendǁlstsq__mutmut_5(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(x, rcond=rcond)

    def xǁNumPyBackendǁlstsq__mutmut_6(self, x: np.ndarray, y: np.ndarray, rcond: float | None) -> tuple:
        return np.linalg.lstsq(x, y, )
    
    xǁNumPyBackendǁlstsq__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁlstsq__mutmut_1': xǁNumPyBackendǁlstsq__mutmut_1, 
        'xǁNumPyBackendǁlstsq__mutmut_2': xǁNumPyBackendǁlstsq__mutmut_2, 
        'xǁNumPyBackendǁlstsq__mutmut_3': xǁNumPyBackendǁlstsq__mutmut_3, 
        'xǁNumPyBackendǁlstsq__mutmut_4': xǁNumPyBackendǁlstsq__mutmut_4, 
        'xǁNumPyBackendǁlstsq__mutmut_5': xǁNumPyBackendǁlstsq__mutmut_5, 
        'xǁNumPyBackendǁlstsq__mutmut_6': xǁNumPyBackendǁlstsq__mutmut_6
    }
    xǁNumPyBackendǁlstsq__mutmut_orig.__name__ = 'xǁNumPyBackendǁlstsq'

    def nanmean(self, x: np.ndarray) -> float:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁnanmean__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁnanmean__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁnanmean__mutmut_orig(self, x: np.ndarray) -> float:
        return float(np.nanmean(x))

    def xǁNumPyBackendǁnanmean__mutmut_1(self, x: np.ndarray) -> float:
        return float(None)

    def xǁNumPyBackendǁnanmean__mutmut_2(self, x: np.ndarray) -> float:
        return float(np.nanmean(None))
    
    xǁNumPyBackendǁnanmean__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁnanmean__mutmut_1': xǁNumPyBackendǁnanmean__mutmut_1, 
        'xǁNumPyBackendǁnanmean__mutmut_2': xǁNumPyBackendǁnanmean__mutmut_2
    }
    xǁNumPyBackendǁnanmean__mutmut_orig.__name__ = 'xǁNumPyBackendǁnanmean'

    def isfinite(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁisfinite__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁisfinite__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁisfinite__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.isfinite(x)

    def xǁNumPyBackendǁisfinite__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.isfinite(None)
    
    xǁNumPyBackendǁisfinite__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁisfinite__mutmut_1': xǁNumPyBackendǁisfinite__mutmut_1
    }
    xǁNumPyBackendǁisfinite__mutmut_orig.__name__ = 'xǁNumPyBackendǁisfinite'

    def errstate(self, **kwargs: Any) -> Any:
        return np.errstate(**kwargs)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁsqrt__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁsqrt__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁsqrt__mutmut_orig(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def xǁNumPyBackendǁsqrt__mutmut_1(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(None)
    
    xǁNumPyBackendǁsqrt__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁsqrt__mutmut_1': xǁNumPyBackendǁsqrt__mutmut_1
    }
    xǁNumPyBackendǁsqrt__mutmut_orig.__name__ = 'xǁNumPyBackendǁsqrt'

    def exp(self, x: np.ndarray | Sequence) -> np.ndarray:
        args = [x]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁexp__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁexp__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁexp__mutmut_orig(self, x: np.ndarray | Sequence) -> np.ndarray:
        """Calculate the exponential of all elements in the input array.

        Args:
            x: Input array or sequence.

        Returns
        -------
            Element-wise exponential of the input array.
        """
        return np.exp(x)

    def xǁNumPyBackendǁexp__mutmut_1(self, x: np.ndarray | Sequence) -> np.ndarray:
        """Calculate the exponential of all elements in the input array.

        Args:
            x: Input array or sequence.

        Returns
        -------
            Element-wise exponential of the input array.
        """
        return np.exp(None)
    
    xǁNumPyBackendǁexp__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁexp__mutmut_1': xǁNumPyBackendǁexp__mutmut_1
    }
    xǁNumPyBackendǁexp__mutmut_orig.__name__ = 'xǁNumPyBackendǁexp'

    def any(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        args = [a, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁany__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁany__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁany__mutmut_orig(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
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

    def xǁNumPyBackendǁany__mutmut_1(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if any element evaluates to True, or array if axis specified.
        """
        return np.any(None, axis=axis)

    def xǁNumPyBackendǁany__mutmut_2(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if any element evaluates to True, or array if axis specified.
        """
        return np.any(a, axis=None)

    def xǁNumPyBackendǁany__mutmut_3(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if any element evaluates to True, or array if axis specified.
        """
        return np.any(axis=axis)

    def xǁNumPyBackendǁany__mutmut_4(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if any element evaluates to True, or array if axis specified.
        """
        return np.any(a, )
    
    xǁNumPyBackendǁany__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁany__mutmut_1': xǁNumPyBackendǁany__mutmut_1, 
        'xǁNumPyBackendǁany__mutmut_2': xǁNumPyBackendǁany__mutmut_2, 
        'xǁNumPyBackendǁany__mutmut_3': xǁNumPyBackendǁany__mutmut_3, 
        'xǁNumPyBackendǁany__mutmut_4': xǁNumPyBackendǁany__mutmut_4
    }
    xǁNumPyBackendǁany__mutmut_orig.__name__ = 'xǁNumPyBackendǁany'

    def all(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        args = [a, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁall__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁall__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁall__mutmut_orig(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
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

    def xǁNumPyBackendǁall__mutmut_1(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if all elements evaluate to True, or array if axis specified.
        """
        return np.all(None, axis=axis)

    def xǁNumPyBackendǁall__mutmut_2(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if all elements evaluate to True, or array if axis specified.
        """
        return np.all(a, axis=None)

    def xǁNumPyBackendǁall__mutmut_3(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if all elements evaluate to True, or array if axis specified.
        """
        return np.all(axis=axis)

    def xǁNumPyBackendǁall__mutmut_4(self, a: np.ndarray, axis: int | tuple | None = None) -> bool | np.ndarray:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            a: Input array.
            axis: Axis or axes along which to operate.
                  If None (default), flattened input is used.

        Returns
        -------
            True if all elements evaluate to True, or array if axis specified.
        """
        return np.all(a, )
    
    xǁNumPyBackendǁall__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁall__mutmut_1': xǁNumPyBackendǁall__mutmut_1, 
        'xǁNumPyBackendǁall__mutmut_2': xǁNumPyBackendǁall__mutmut_2, 
        'xǁNumPyBackendǁall__mutmut_3': xǁNumPyBackendǁall__mutmut_3, 
        'xǁNumPyBackendǁall__mutmut_4': xǁNumPyBackendǁall__mutmut_4
    }
    xǁNumPyBackendǁall__mutmut_orig.__name__ = 'xǁNumPyBackendǁall'

    def squeeze(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
        args = [a, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁsqueeze__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁsqueeze__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁsqueeze__mutmut_orig(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
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

    def xǁNumPyBackendǁsqueeze__mutmut_1(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array.

        Args:
            a: Input array.
            axis: Selects subset of single-dimensional entries in the shape.
                  If None (default), squeezes all single-dimensional entries.

        Returns
        -------
            Squeezed array with specified dimensions removed.
        """
        return np.squeeze(None, axis=axis)

    def xǁNumPyBackendǁsqueeze__mutmut_2(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array.

        Args:
            a: Input array.
            axis: Selects subset of single-dimensional entries in the shape.
                  If None (default), squeezes all single-dimensional entries.

        Returns
        -------
            Squeezed array with specified dimensions removed.
        """
        return np.squeeze(a, axis=None)

    def xǁNumPyBackendǁsqueeze__mutmut_3(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array.

        Args:
            a: Input array.
            axis: Selects subset of single-dimensional entries in the shape.
                  If None (default), squeezes all single-dimensional entries.

        Returns
        -------
            Squeezed array with specified dimensions removed.
        """
        return np.squeeze(axis=axis)

    def xǁNumPyBackendǁsqueeze__mutmut_4(self, a: np.ndarray, axis: int | tuple | None = None) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array.

        Args:
            a: Input array.
            axis: Selects subset of single-dimensional entries in the shape.
                  If None (default), squeezes all single-dimensional entries.

        Returns
        -------
            Squeezed array with specified dimensions removed.
        """
        return np.squeeze(a, )
    
    xǁNumPyBackendǁsqueeze__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁsqueeze__mutmut_1': xǁNumPyBackendǁsqueeze__mutmut_1, 
        'xǁNumPyBackendǁsqueeze__mutmut_2': xǁNumPyBackendǁsqueeze__mutmut_2, 
        'xǁNumPyBackendǁsqueeze__mutmut_3': xǁNumPyBackendǁsqueeze__mutmut_3, 
        'xǁNumPyBackendǁsqueeze__mutmut_4': xǁNumPyBackendǁsqueeze__mutmut_4
    }
    xǁNumPyBackendǁsqueeze__mutmut_orig.__name__ = 'xǁNumPyBackendǁsqueeze'

    def repeat(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
        args = [a, repeats, axis]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁrepeat__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁrepeat__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁrepeat__mutmut_orig(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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

    def xǁNumPyBackendǁrepeat__mutmut_1(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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
        return np.repeat(None, repeats, axis=axis)

    def xǁNumPyBackendǁrepeat__mutmut_2(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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
        return np.repeat(a, None, axis=axis)

    def xǁNumPyBackendǁrepeat__mutmut_3(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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
        return np.repeat(a, repeats, axis=None)

    def xǁNumPyBackendǁrepeat__mutmut_4(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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
        return np.repeat(repeats, axis=axis)

    def xǁNumPyBackendǁrepeat__mutmut_5(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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
        return np.repeat(a, axis=axis)

    def xǁNumPyBackendǁrepeat__mutmut_6(self, a: np.ndarray, repeats: int | Sequence, axis: int | None = None) -> np.ndarray:
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
        return np.repeat(a, repeats, )
    
    xǁNumPyBackendǁrepeat__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁrepeat__mutmut_1': xǁNumPyBackendǁrepeat__mutmut_1, 
        'xǁNumPyBackendǁrepeat__mutmut_2': xǁNumPyBackendǁrepeat__mutmut_2, 
        'xǁNumPyBackendǁrepeat__mutmut_3': xǁNumPyBackendǁrepeat__mutmut_3, 
        'xǁNumPyBackendǁrepeat__mutmut_4': xǁNumPyBackendǁrepeat__mutmut_4, 
        'xǁNumPyBackendǁrepeat__mutmut_5': xǁNumPyBackendǁrepeat__mutmut_5, 
        'xǁNumPyBackendǁrepeat__mutmut_6': xǁNumPyBackendǁrepeat__mutmut_6
    }
    xǁNumPyBackendǁrepeat__mutmut_orig.__name__ = 'xǁNumPyBackendǁrepeat'

    def power(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        args = [x, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁpower__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁpower__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁpower__mutmut_orig(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return np.power(x, y)

    def xǁNumPyBackendǁpower__mutmut_1(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return np.power(None, y)

    def xǁNumPyBackendǁpower__mutmut_2(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return np.power(x, None)

    def xǁNumPyBackendǁpower__mutmut_3(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return np.power(y)

    def xǁNumPyBackendǁpower__mutmut_4(self, x: np.ndarray | Sequence, y: float | np.ndarray) -> np.ndarray:
        """First array elements raised to powers from second array.

        Args:
            x: Base array or scalar.
            y: Exponent array or scalar.

        Returns
        -------
            Array with elements of x raised to the corresponding powers of y.
        """
        return np.power(x, )
    
    xǁNumPyBackendǁpower__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁpower__mutmut_1': xǁNumPyBackendǁpower__mutmut_1, 
        'xǁNumPyBackendǁpower__mutmut_2': xǁNumPyBackendǁpower__mutmut_2, 
        'xǁNumPyBackendǁpower__mutmut_3': xǁNumPyBackendǁpower__mutmut_3, 
        'xǁNumPyBackendǁpower__mutmut_4': xǁNumPyBackendǁpower__mutmut_4
    }
    xǁNumPyBackendǁpower__mutmut_orig.__name__ = 'xǁNumPyBackendǁpower'

    def ones_like(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        args = [a, dtype, subok, shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁones_like__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁones_like__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁones_like__mutmut_orig(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_1(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = False,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_2(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(None, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_3(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=None, subok=subok, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_4(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=dtype, subok=None, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_5(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=dtype, subok=subok, shape=None)

    def xǁNumPyBackendǁones_like__mutmut_6(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_7(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, subok=subok, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_8(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=dtype, shape=shape)

    def xǁNumPyBackendǁones_like__mutmut_9(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of ones with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of ones with same shape and type as input array.
        """
        return np.ones_like(a, dtype=dtype, subok=subok, )
    
    xǁNumPyBackendǁones_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁones_like__mutmut_1': xǁNumPyBackendǁones_like__mutmut_1, 
        'xǁNumPyBackendǁones_like__mutmut_2': xǁNumPyBackendǁones_like__mutmut_2, 
        'xǁNumPyBackendǁones_like__mutmut_3': xǁNumPyBackendǁones_like__mutmut_3, 
        'xǁNumPyBackendǁones_like__mutmut_4': xǁNumPyBackendǁones_like__mutmut_4, 
        'xǁNumPyBackendǁones_like__mutmut_5': xǁNumPyBackendǁones_like__mutmut_5, 
        'xǁNumPyBackendǁones_like__mutmut_6': xǁNumPyBackendǁones_like__mutmut_6, 
        'xǁNumPyBackendǁones_like__mutmut_7': xǁNumPyBackendǁones_like__mutmut_7, 
        'xǁNumPyBackendǁones_like__mutmut_8': xǁNumPyBackendǁones_like__mutmut_8, 
        'xǁNumPyBackendǁones_like__mutmut_9': xǁNumPyBackendǁones_like__mutmut_9
    }
    xǁNumPyBackendǁones_like__mutmut_orig.__name__ = 'xǁNumPyBackendǁones_like'

    def zeros_like(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        args = [a, dtype, subok, shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁzeros_like__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁzeros_like__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁzeros_like__mutmut_orig(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_1(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = False,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_2(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(None, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_3(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=None, subok=subok, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_4(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=dtype, subok=None, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_5(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=dtype, subok=subok, shape=None)

    def xǁNumPyBackendǁzeros_like__mutmut_6(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_7(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, subok=subok, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_8(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=dtype, shape=shape)

    def xǁNumPyBackendǁzeros_like__mutmut_9(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        """Return an array of zeros with the same shape and type as a given array.

        Args:
            a: Input array to model the shape and type of the output.
            dtype: Overrides the data type of the result.
            subok: If True, subclasses will be passed through.
            shape: Overrides the shape of the result.

        Returns
        -------
            Array of zeros with same shape and type as input array.
        """
        return np.zeros_like(a, dtype=dtype, subok=subok, )
    
    xǁNumPyBackendǁzeros_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁzeros_like__mutmut_1': xǁNumPyBackendǁzeros_like__mutmut_1, 
        'xǁNumPyBackendǁzeros_like__mutmut_2': xǁNumPyBackendǁzeros_like__mutmut_2, 
        'xǁNumPyBackendǁzeros_like__mutmut_3': xǁNumPyBackendǁzeros_like__mutmut_3, 
        'xǁNumPyBackendǁzeros_like__mutmut_4': xǁNumPyBackendǁzeros_like__mutmut_4, 
        'xǁNumPyBackendǁzeros_like__mutmut_5': xǁNumPyBackendǁzeros_like__mutmut_5, 
        'xǁNumPyBackendǁzeros_like__mutmut_6': xǁNumPyBackendǁzeros_like__mutmut_6, 
        'xǁNumPyBackendǁzeros_like__mutmut_7': xǁNumPyBackendǁzeros_like__mutmut_7, 
        'xǁNumPyBackendǁzeros_like__mutmut_8': xǁNumPyBackendǁzeros_like__mutmut_8, 
        'xǁNumPyBackendǁzeros_like__mutmut_9': xǁNumPyBackendǁzeros_like__mutmut_9
    }
    xǁNumPyBackendǁzeros_like__mutmut_orig.__name__ = 'xǁNumPyBackendǁzeros_like'

    def empty_like(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        args = [a, dtype, subok, shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁempty_like__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁempty_like__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁempty_like__mutmut_orig(
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

    def xǁNumPyBackendǁempty_like__mutmut_1(
        self,
        a: np.ndarray | Sequence,
        dtype: type | None = None,
        subok: bool = False,
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

    def xǁNumPyBackendǁempty_like__mutmut_2(
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
        return np.empty_like(None, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁempty_like__mutmut_3(
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
        return np.empty_like(a, dtype=None, subok=subok, shape=shape)

    def xǁNumPyBackendǁempty_like__mutmut_4(
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
        return np.empty_like(a, dtype=dtype, subok=None, shape=shape)

    def xǁNumPyBackendǁempty_like__mutmut_5(
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
        return np.empty_like(a, dtype=dtype, subok=subok, shape=None)

    def xǁNumPyBackendǁempty_like__mutmut_6(
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
        return np.empty_like(dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁempty_like__mutmut_7(
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
        return np.empty_like(a, subok=subok, shape=shape)

    def xǁNumPyBackendǁempty_like__mutmut_8(
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
        return np.empty_like(a, dtype=dtype, shape=shape)

    def xǁNumPyBackendǁempty_like__mutmut_9(
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
        return np.empty_like(a, dtype=dtype, subok=subok, )
    
    xǁNumPyBackendǁempty_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁempty_like__mutmut_1': xǁNumPyBackendǁempty_like__mutmut_1, 
        'xǁNumPyBackendǁempty_like__mutmut_2': xǁNumPyBackendǁempty_like__mutmut_2, 
        'xǁNumPyBackendǁempty_like__mutmut_3': xǁNumPyBackendǁempty_like__mutmut_3, 
        'xǁNumPyBackendǁempty_like__mutmut_4': xǁNumPyBackendǁempty_like__mutmut_4, 
        'xǁNumPyBackendǁempty_like__mutmut_5': xǁNumPyBackendǁempty_like__mutmut_5, 
        'xǁNumPyBackendǁempty_like__mutmut_6': xǁNumPyBackendǁempty_like__mutmut_6, 
        'xǁNumPyBackendǁempty_like__mutmut_7': xǁNumPyBackendǁempty_like__mutmut_7, 
        'xǁNumPyBackendǁempty_like__mutmut_8': xǁNumPyBackendǁempty_like__mutmut_8, 
        'xǁNumPyBackendǁempty_like__mutmut_9': xǁNumPyBackendǁempty_like__mutmut_9
    }
    xǁNumPyBackendǁempty_like__mutmut_orig.__name__ = 'xǁNumPyBackendǁempty_like'

    def full_like(
        self,
        a: np.ndarray | Sequence,
        fill_value: int | float,
        dtype: type | None = None,
        subok: bool = True,
        shape: int | Sequence | None = None,
    ) -> np.ndarray:
        args = [a, fill_value, dtype, subok, shape]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNumPyBackendǁfull_like__mutmut_orig'), object.__getattribute__(self, 'xǁNumPyBackendǁfull_like__mutmut_mutants'), args, kwargs, self)

    def xǁNumPyBackendǁfull_like__mutmut_orig(
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

    def xǁNumPyBackendǁfull_like__mutmut_1(
        self,
        a: np.ndarray | Sequence,
        fill_value: int | float,
        dtype: type | None = None,
        subok: bool = False,
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

    def xǁNumPyBackendǁfull_like__mutmut_2(
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
        return np.full_like(None, fill_value, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_3(
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
        return np.full_like(a, None, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_4(
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
        return np.full_like(a, fill_value, dtype=None, subok=subok, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_5(
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
        return np.full_like(a, fill_value, dtype=dtype, subok=None, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_6(
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
        return np.full_like(a, fill_value, dtype=dtype, subok=subok, shape=None)

    def xǁNumPyBackendǁfull_like__mutmut_7(
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
        return np.full_like(fill_value, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_8(
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
        return np.full_like(a, dtype=dtype, subok=subok, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_9(
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
        return np.full_like(a, fill_value, subok=subok, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_10(
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
        return np.full_like(a, fill_value, dtype=dtype, shape=shape)

    def xǁNumPyBackendǁfull_like__mutmut_11(
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
        return np.full_like(a, fill_value, dtype=dtype, subok=subok, )
    
    xǁNumPyBackendǁfull_like__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNumPyBackendǁfull_like__mutmut_1': xǁNumPyBackendǁfull_like__mutmut_1, 
        'xǁNumPyBackendǁfull_like__mutmut_2': xǁNumPyBackendǁfull_like__mutmut_2, 
        'xǁNumPyBackendǁfull_like__mutmut_3': xǁNumPyBackendǁfull_like__mutmut_3, 
        'xǁNumPyBackendǁfull_like__mutmut_4': xǁNumPyBackendǁfull_like__mutmut_4, 
        'xǁNumPyBackendǁfull_like__mutmut_5': xǁNumPyBackendǁfull_like__mutmut_5, 
        'xǁNumPyBackendǁfull_like__mutmut_6': xǁNumPyBackendǁfull_like__mutmut_6, 
        'xǁNumPyBackendǁfull_like__mutmut_7': xǁNumPyBackendǁfull_like__mutmut_7, 
        'xǁNumPyBackendǁfull_like__mutmut_8': xǁNumPyBackendǁfull_like__mutmut_8, 
        'xǁNumPyBackendǁfull_like__mutmut_9': xǁNumPyBackendǁfull_like__mutmut_9, 
        'xǁNumPyBackendǁfull_like__mutmut_10': xǁNumPyBackendǁfull_like__mutmut_10, 
        'xǁNumPyBackendǁfull_like__mutmut_11': xǁNumPyBackendǁfull_like__mutmut_11
    }
    xǁNumPyBackendǁfull_like__mutmut_orig.__name__ = 'xǁNumPyBackendǁfull_like'
