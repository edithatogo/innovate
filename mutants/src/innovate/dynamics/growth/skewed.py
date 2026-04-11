from innovate.backend import current_backend as B

from .base import GrowthCurve
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


class SkewedGrowth(GrowthCurve):
    """Models asymmetric S-shaped growth where the rate of adoption is not
    symmetric around the inflection point. The inflection point is typically
    earlier than 50% of the market potential (around 37%), leading to a
    growth phase that decelerates more slowly than it accelerates. This is
    often referred to as the Gompertz growth model.

    Core Behavior: Represents growth with diminishing returns to scale or
    a rapid initial uptake followed by a long tail of adoption.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        args = [current_adopters, total_potential]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSkewedGrowthǁcompute_growth_rate__mutmut_orig'), object.__getattribute__(self, 'xǁSkewedGrowthǁcompute_growth_rate__mutmut_mutants'), args, kwargs, self)

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_orig(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_1(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = None
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_2(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = None
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_3(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[1] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_4(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(None, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_5(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, None) else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_6(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr("__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_7(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, ) else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_8(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "XX__len__XX") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_9(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__LEN__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_10(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = None

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_11(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get(None, 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_12(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", None)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_13(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get(0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_14(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", )

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_15(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("XXshape_cXX", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_16(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("SHAPE_C", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_17(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 1.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_18(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 and N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_19(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K < 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_20(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 1 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_21(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N < 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_22(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 1:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_23(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 1

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_24(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N / (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_25(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c / N * (B.log(K) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_26(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) + B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_27(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(None) - B.log(N))

    def xǁSkewedGrowthǁcompute_growth_rate__mutmut_28(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, "__len__") else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0

        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(None))
    
    xǁSkewedGrowthǁcompute_growth_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSkewedGrowthǁcompute_growth_rate__mutmut_1': xǁSkewedGrowthǁcompute_growth_rate__mutmut_1, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_2': xǁSkewedGrowthǁcompute_growth_rate__mutmut_2, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_3': xǁSkewedGrowthǁcompute_growth_rate__mutmut_3, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_4': xǁSkewedGrowthǁcompute_growth_rate__mutmut_4, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_5': xǁSkewedGrowthǁcompute_growth_rate__mutmut_5, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_6': xǁSkewedGrowthǁcompute_growth_rate__mutmut_6, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_7': xǁSkewedGrowthǁcompute_growth_rate__mutmut_7, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_8': xǁSkewedGrowthǁcompute_growth_rate__mutmut_8, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_9': xǁSkewedGrowthǁcompute_growth_rate__mutmut_9, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_10': xǁSkewedGrowthǁcompute_growth_rate__mutmut_10, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_11': xǁSkewedGrowthǁcompute_growth_rate__mutmut_11, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_12': xǁSkewedGrowthǁcompute_growth_rate__mutmut_12, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_13': xǁSkewedGrowthǁcompute_growth_rate__mutmut_13, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_14': xǁSkewedGrowthǁcompute_growth_rate__mutmut_14, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_15': xǁSkewedGrowthǁcompute_growth_rate__mutmut_15, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_16': xǁSkewedGrowthǁcompute_growth_rate__mutmut_16, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_17': xǁSkewedGrowthǁcompute_growth_rate__mutmut_17, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_18': xǁSkewedGrowthǁcompute_growth_rate__mutmut_18, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_19': xǁSkewedGrowthǁcompute_growth_rate__mutmut_19, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_20': xǁSkewedGrowthǁcompute_growth_rate__mutmut_20, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_21': xǁSkewedGrowthǁcompute_growth_rate__mutmut_21, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_22': xǁSkewedGrowthǁcompute_growth_rate__mutmut_22, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_23': xǁSkewedGrowthǁcompute_growth_rate__mutmut_23, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_24': xǁSkewedGrowthǁcompute_growth_rate__mutmut_24, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_25': xǁSkewedGrowthǁcompute_growth_rate__mutmut_25, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_26': xǁSkewedGrowthǁcompute_growth_rate__mutmut_26, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_27': xǁSkewedGrowthǁcompute_growth_rate__mutmut_27, 
        'xǁSkewedGrowthǁcompute_growth_rate__mutmut_28': xǁSkewedGrowthǁcompute_growth_rate__mutmut_28
    }
    xǁSkewedGrowthǁcompute_growth_rate__mutmut_orig.__name__ = 'xǁSkewedGrowthǁcompute_growth_rate'

    def predict_cumulative(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        args = [time_points, initial_adopters, total_potential]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSkewedGrowthǁpredict_cumulative__mutmut_orig'), object.__getattribute__(self, 'xǁSkewedGrowthǁpredict_cumulative__mutmut_mutants'), args, kwargs, self)

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_orig(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_1(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = None
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_2(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = None
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_3(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get(None, 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_4(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", None)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_5(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get(1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_6(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", )
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_7(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("XXshape_bXX", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_8(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("SHAPE_B", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_9(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 2.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_10(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = None

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_11(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get(None, 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_12(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", None)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_13(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get(0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_14(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", )

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_15(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("XXshape_cXX", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_16(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("SHAPE_C", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_17(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 1.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_18(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K / B.exp(-b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_19(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(None)

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_20(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b / B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_21(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(+b * B.exp(-c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_22(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(None))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_23(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c / B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_24(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(+c * B.array(time_points)))

    def xǁSkewedGrowthǁpredict_cumulative__mutmut_25(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.

        Parameters
        ----------
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).

        Returns
        -------
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(None)))
    
    xǁSkewedGrowthǁpredict_cumulative__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSkewedGrowthǁpredict_cumulative__mutmut_1': xǁSkewedGrowthǁpredict_cumulative__mutmut_1, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_2': xǁSkewedGrowthǁpredict_cumulative__mutmut_2, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_3': xǁSkewedGrowthǁpredict_cumulative__mutmut_3, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_4': xǁSkewedGrowthǁpredict_cumulative__mutmut_4, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_5': xǁSkewedGrowthǁpredict_cumulative__mutmut_5, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_6': xǁSkewedGrowthǁpredict_cumulative__mutmut_6, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_7': xǁSkewedGrowthǁpredict_cumulative__mutmut_7, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_8': xǁSkewedGrowthǁpredict_cumulative__mutmut_8, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_9': xǁSkewedGrowthǁpredict_cumulative__mutmut_9, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_10': xǁSkewedGrowthǁpredict_cumulative__mutmut_10, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_11': xǁSkewedGrowthǁpredict_cumulative__mutmut_11, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_12': xǁSkewedGrowthǁpredict_cumulative__mutmut_12, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_13': xǁSkewedGrowthǁpredict_cumulative__mutmut_13, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_14': xǁSkewedGrowthǁpredict_cumulative__mutmut_14, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_15': xǁSkewedGrowthǁpredict_cumulative__mutmut_15, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_16': xǁSkewedGrowthǁpredict_cumulative__mutmut_16, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_17': xǁSkewedGrowthǁpredict_cumulative__mutmut_17, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_18': xǁSkewedGrowthǁpredict_cumulative__mutmut_18, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_19': xǁSkewedGrowthǁpredict_cumulative__mutmut_19, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_20': xǁSkewedGrowthǁpredict_cumulative__mutmut_20, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_21': xǁSkewedGrowthǁpredict_cumulative__mutmut_21, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_22': xǁSkewedGrowthǁpredict_cumulative__mutmut_22, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_23': xǁSkewedGrowthǁpredict_cumulative__mutmut_23, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_24': xǁSkewedGrowthǁpredict_cumulative__mutmut_24, 
        'xǁSkewedGrowthǁpredict_cumulative__mutmut_25': xǁSkewedGrowthǁpredict_cumulative__mutmut_25
    }
    xǁSkewedGrowthǁpredict_cumulative__mutmut_orig.__name__ = 'xǁSkewedGrowthǁpredict_cumulative'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSkewedGrowthǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁSkewedGrowthǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "XXshape_bXX": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "SHAPE_B": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "XXtypeXX": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "TYPE": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "XXfloatXX",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "FLOAT",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "XXdefaultXX": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "DEFAULT": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 2.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "XXdescriptionXX": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "DESCRIPTION": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "XXShape parameter b.XX",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "SHAPE PARAMETER B.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "XXshape_cXX": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "SHAPE_C": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "XXtypeXX": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "TYPE": "float",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "XXfloatXX",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "FLOAT",
                "default": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "XXdefaultXX": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "DEFAULT": 0.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 1.1,
                "description": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "XXdescriptionXX": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "DESCRIPTION": "Shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "XXShape parameter c.XX",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "shape parameter c.",
            },
        }

    def xǁSkewedGrowthǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.

        Returns
        -------
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b.",
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "SHAPE PARAMETER C.",
            },
        }
    
    xǁSkewedGrowthǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSkewedGrowthǁget_parameters_schema__mutmut_1': xǁSkewedGrowthǁget_parameters_schema__mutmut_1, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_2': xǁSkewedGrowthǁget_parameters_schema__mutmut_2, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_3': xǁSkewedGrowthǁget_parameters_schema__mutmut_3, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_4': xǁSkewedGrowthǁget_parameters_schema__mutmut_4, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_5': xǁSkewedGrowthǁget_parameters_schema__mutmut_5, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_6': xǁSkewedGrowthǁget_parameters_schema__mutmut_6, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_7': xǁSkewedGrowthǁget_parameters_schema__mutmut_7, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_8': xǁSkewedGrowthǁget_parameters_schema__mutmut_8, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_9': xǁSkewedGrowthǁget_parameters_schema__mutmut_9, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_10': xǁSkewedGrowthǁget_parameters_schema__mutmut_10, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_11': xǁSkewedGrowthǁget_parameters_schema__mutmut_11, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_12': xǁSkewedGrowthǁget_parameters_schema__mutmut_12, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_13': xǁSkewedGrowthǁget_parameters_schema__mutmut_13, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_14': xǁSkewedGrowthǁget_parameters_schema__mutmut_14, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_15': xǁSkewedGrowthǁget_parameters_schema__mutmut_15, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_16': xǁSkewedGrowthǁget_parameters_schema__mutmut_16, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_17': xǁSkewedGrowthǁget_parameters_schema__mutmut_17, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_18': xǁSkewedGrowthǁget_parameters_schema__mutmut_18, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_19': xǁSkewedGrowthǁget_parameters_schema__mutmut_19, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_20': xǁSkewedGrowthǁget_parameters_schema__mutmut_20, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_21': xǁSkewedGrowthǁget_parameters_schema__mutmut_21, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_22': xǁSkewedGrowthǁget_parameters_schema__mutmut_22, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_23': xǁSkewedGrowthǁget_parameters_schema__mutmut_23, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_24': xǁSkewedGrowthǁget_parameters_schema__mutmut_24, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_25': xǁSkewedGrowthǁget_parameters_schema__mutmut_25, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_26': xǁSkewedGrowthǁget_parameters_schema__mutmut_26, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_27': xǁSkewedGrowthǁget_parameters_schema__mutmut_27, 
        'xǁSkewedGrowthǁget_parameters_schema__mutmut_28': xǁSkewedGrowthǁget_parameters_schema__mutmut_28
    }
    xǁSkewedGrowthǁget_parameters_schema__mutmut_orig.__name__ = 'xǁSkewedGrowthǁget_parameters_schema'
