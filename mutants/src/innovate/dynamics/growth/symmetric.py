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


class SymmetricGrowth(GrowthCurve):
    """Models symmetric S-shaped growth where the rate of adoption is proportional
    to both the number of adopters and the remaining potential adopters. The
    inflection point is at 50% of the market potential. This is often referred
    to as the Logistic growth model.

    Core Behavior: Growth is driven by internal imitation or simple resource
    constraints. It's a good baseline for simple, internally-driven diffusion.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        args = [current_adopters, total_potential]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_orig'), object.__getattribute__(self, 'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_mutants'), args, kwargs, self)

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_orig(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_1(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = None
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_2(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get(None, 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_3(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", None)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_4(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get(0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_5(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", )
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_6(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("XXgrowth_rateXX", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_7(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("GROWTH_RATE", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_8(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 1.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_9(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = None
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_10(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = None
        return r * N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_11(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N / (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_12(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r / N * (1 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_13(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 + N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_14(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (2 - N / K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_15(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N * K) if K > 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_16(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K >= 0 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_17(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 1 else 0

    def xǁSymmetricGrowthǁcompute_growth_rate__mutmut_18(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters
        ----------
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns
        -------
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 1
    
    xǁSymmetricGrowthǁcompute_growth_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_1': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_1, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_2': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_2, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_3': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_3, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_4': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_4, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_5': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_5, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_6': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_6, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_7': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_7, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_8': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_8, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_9': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_9, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_10': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_10, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_11': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_11, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_12': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_12, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_13': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_13, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_14': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_14, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_15': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_15, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_16': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_16, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_17': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_17, 
        'xǁSymmetricGrowthǁcompute_growth_rate__mutmut_18': xǁSymmetricGrowthǁcompute_growth_rate__mutmut_18
    }
    xǁSymmetricGrowthǁcompute_growth_rate__mutmut_orig.__name__ = 'xǁSymmetricGrowthǁcompute_growth_rate'

    def predict_cumulative(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        args = [time_points, initial_adopters, total_potential]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSymmetricGrowthǁpredict_cumulative__mutmut_orig'), object.__getattribute__(self, 'xǁSymmetricGrowthǁpredict_cumulative__mutmut_mutants'), args, kwargs, self)

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_orig(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_1(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = None
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_2(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get(None, 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_3(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", None)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_4(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get(0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_5(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", )
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_6(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("XXgrowth_rateXX", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_7(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("GROWTH_RATE", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_8(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 1.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_9(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = None

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_10(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(None, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_11(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, None, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_12(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=None)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_13(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_14(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_15(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_16(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = None
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_17(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_18(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            None,
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_19(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_20(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_21(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method=None,
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_22(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_23(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_24(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_25(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_26(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_27(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_28(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_29(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_30(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.flatten()

    def xǁSymmetricGrowthǁpredict_cumulative__mutmut_31(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns
        -------
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.flatten()
    
    xǁSymmetricGrowthǁpredict_cumulative__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSymmetricGrowthǁpredict_cumulative__mutmut_1': xǁSymmetricGrowthǁpredict_cumulative__mutmut_1, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_2': xǁSymmetricGrowthǁpredict_cumulative__mutmut_2, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_3': xǁSymmetricGrowthǁpredict_cumulative__mutmut_3, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_4': xǁSymmetricGrowthǁpredict_cumulative__mutmut_4, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_5': xǁSymmetricGrowthǁpredict_cumulative__mutmut_5, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_6': xǁSymmetricGrowthǁpredict_cumulative__mutmut_6, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_7': xǁSymmetricGrowthǁpredict_cumulative__mutmut_7, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_8': xǁSymmetricGrowthǁpredict_cumulative__mutmut_8, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_9': xǁSymmetricGrowthǁpredict_cumulative__mutmut_9, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_10': xǁSymmetricGrowthǁpredict_cumulative__mutmut_10, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_11': xǁSymmetricGrowthǁpredict_cumulative__mutmut_11, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_12': xǁSymmetricGrowthǁpredict_cumulative__mutmut_12, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_13': xǁSymmetricGrowthǁpredict_cumulative__mutmut_13, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_14': xǁSymmetricGrowthǁpredict_cumulative__mutmut_14, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_15': xǁSymmetricGrowthǁpredict_cumulative__mutmut_15, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_16': xǁSymmetricGrowthǁpredict_cumulative__mutmut_16, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_17': xǁSymmetricGrowthǁpredict_cumulative__mutmut_17, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_18': xǁSymmetricGrowthǁpredict_cumulative__mutmut_18, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_19': xǁSymmetricGrowthǁpredict_cumulative__mutmut_19, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_20': xǁSymmetricGrowthǁpredict_cumulative__mutmut_20, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_21': xǁSymmetricGrowthǁpredict_cumulative__mutmut_21, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_22': xǁSymmetricGrowthǁpredict_cumulative__mutmut_22, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_23': xǁSymmetricGrowthǁpredict_cumulative__mutmut_23, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_24': xǁSymmetricGrowthǁpredict_cumulative__mutmut_24, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_25': xǁSymmetricGrowthǁpredict_cumulative__mutmut_25, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_26': xǁSymmetricGrowthǁpredict_cumulative__mutmut_26, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_27': xǁSymmetricGrowthǁpredict_cumulative__mutmut_27, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_28': xǁSymmetricGrowthǁpredict_cumulative__mutmut_28, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_29': xǁSymmetricGrowthǁpredict_cumulative__mutmut_29, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_30': xǁSymmetricGrowthǁpredict_cumulative__mutmut_30, 
        'xǁSymmetricGrowthǁpredict_cumulative__mutmut_31': xǁSymmetricGrowthǁpredict_cumulative__mutmut_31
    }
    xǁSymmetricGrowthǁpredict_cumulative__mutmut_orig.__name__ = 'xǁSymmetricGrowthǁpredict_cumulative'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSymmetricGrowthǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁSymmetricGrowthǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "XXgrowth_rateXX": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "GROWTH_RATE": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "XXtypeXX": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "TYPE": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "XXfloatXX",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "FLOAT",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "XXdefaultXX": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "DEFAULT": 0.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 1.1,
                "description": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "XXdescriptionXX": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "DESCRIPTION": "The intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "description": "XXThe intrinsic growth rate.XX",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "description": "the intrinsic growth rate.",
            },
        }

    def xǁSymmetricGrowthǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "description": "THE INTRINSIC GROWTH RATE.",
            },
        }
    
    xǁSymmetricGrowthǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSymmetricGrowthǁget_parameters_schema__mutmut_1': xǁSymmetricGrowthǁget_parameters_schema__mutmut_1, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_2': xǁSymmetricGrowthǁget_parameters_schema__mutmut_2, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_3': xǁSymmetricGrowthǁget_parameters_schema__mutmut_3, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_4': xǁSymmetricGrowthǁget_parameters_schema__mutmut_4, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_5': xǁSymmetricGrowthǁget_parameters_schema__mutmut_5, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_6': xǁSymmetricGrowthǁget_parameters_schema__mutmut_6, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_7': xǁSymmetricGrowthǁget_parameters_schema__mutmut_7, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_8': xǁSymmetricGrowthǁget_parameters_schema__mutmut_8, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_9': xǁSymmetricGrowthǁget_parameters_schema__mutmut_9, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_10': xǁSymmetricGrowthǁget_parameters_schema__mutmut_10, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_11': xǁSymmetricGrowthǁget_parameters_schema__mutmut_11, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_12': xǁSymmetricGrowthǁget_parameters_schema__mutmut_12, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_13': xǁSymmetricGrowthǁget_parameters_schema__mutmut_13, 
        'xǁSymmetricGrowthǁget_parameters_schema__mutmut_14': xǁSymmetricGrowthǁget_parameters_schema__mutmut_14
    }
    xǁSymmetricGrowthǁget_parameters_schema__mutmut_orig.__name__ = 'xǁSymmetricGrowthǁget_parameters_schema'
