from innovate.backend import current_backend as B

from .base import CompetitiveInteraction
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


class ReplicatorDynamics(CompetitiveInteraction):
    """Models the evolution of strategy proportions based on relative fitness/payoff in a game."""

    def compute_interaction_rates(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_orig'), object.__getattribute__(self, 'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_mutants'), args, kwargs, self)

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_orig(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_1(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = None
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_2(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get(None)
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_3(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("XXxXX")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_4(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("X")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_5(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = None

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_6(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get(None)

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_7(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("XXpayoff_matrixXX")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_8(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("PAYOFF_MATRIX")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_9(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = None
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_10(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(None, B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_11(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), None)
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_12(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_13(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), )
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_14(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(None), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_15(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(None))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_16(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = None

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_17(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(None)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_18(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) / U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_19(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(None) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_20(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = None
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_21(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) / (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_22(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(None) * (U - U_bar)
        return dxdt

    def xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_23(self, **params):
        """Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters
        ----------
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns
        -------
                array: The rate of change of each strategy's proportion.
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U + U_bar)
        return dxdt
    
    xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_1': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_1, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_2': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_2, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_3': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_3, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_4': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_4, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_5': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_5, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_6': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_6, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_7': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_7, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_8': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_8, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_9': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_9, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_10': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_10, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_11': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_11, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_12': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_12, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_13': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_13, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_14': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_14, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_15': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_15, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_16': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_16, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_17': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_17, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_18': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_18, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_19': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_19, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_20': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_20, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_21': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_21, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_22': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_22, 
        'xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_23': xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_23
    }
    xǁReplicatorDynamicsǁcompute_interaction_rates__mutmut_orig.__name__ = 'xǁReplicatorDynamicsǁcompute_interaction_rates'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁReplicatorDynamicsǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁReplicatorDynamicsǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁReplicatorDynamicsǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = None
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get(None, [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", None)
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get([])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", )
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("XXx0XX", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("X0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError(None)

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("XXInitial proportions must be provided.XX")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("INITIAL PROPORTIONS MUST BE PROVIDED.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(**params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = None
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            None,
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁReplicatorDynamicsǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns
        -------
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises
        ------
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁReplicatorDynamicsǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁReplicatorDynamicsǁpredict_states__mutmut_1': xǁReplicatorDynamicsǁpredict_states__mutmut_1, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_2': xǁReplicatorDynamicsǁpredict_states__mutmut_2, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_3': xǁReplicatorDynamicsǁpredict_states__mutmut_3, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_4': xǁReplicatorDynamicsǁpredict_states__mutmut_4, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_5': xǁReplicatorDynamicsǁpredict_states__mutmut_5, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_6': xǁReplicatorDynamicsǁpredict_states__mutmut_6, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_7': xǁReplicatorDynamicsǁpredict_states__mutmut_7, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_8': xǁReplicatorDynamicsǁpredict_states__mutmut_8, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_9': xǁReplicatorDynamicsǁpredict_states__mutmut_9, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_10': xǁReplicatorDynamicsǁpredict_states__mutmut_10, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_11': xǁReplicatorDynamicsǁpredict_states__mutmut_11, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_12': xǁReplicatorDynamicsǁpredict_states__mutmut_12, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_13': xǁReplicatorDynamicsǁpredict_states__mutmut_13, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_14': xǁReplicatorDynamicsǁpredict_states__mutmut_14, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_15': xǁReplicatorDynamicsǁpredict_states__mutmut_15, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_16': xǁReplicatorDynamicsǁpredict_states__mutmut_16, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_17': xǁReplicatorDynamicsǁpredict_states__mutmut_17, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_18': xǁReplicatorDynamicsǁpredict_states__mutmut_18, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_19': xǁReplicatorDynamicsǁpredict_states__mutmut_19, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_20': xǁReplicatorDynamicsǁpredict_states__mutmut_20, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_21': xǁReplicatorDynamicsǁpredict_states__mutmut_21, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_22': xǁReplicatorDynamicsǁpredict_states__mutmut_22, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_23': xǁReplicatorDynamicsǁpredict_states__mutmut_23, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_24': xǁReplicatorDynamicsǁpredict_states__mutmut_24, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_25': xǁReplicatorDynamicsǁpredict_states__mutmut_25, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_26': xǁReplicatorDynamicsǁpredict_states__mutmut_26, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_27': xǁReplicatorDynamicsǁpredict_states__mutmut_27, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_28': xǁReplicatorDynamicsǁpredict_states__mutmut_28, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_29': xǁReplicatorDynamicsǁpredict_states__mutmut_29, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_30': xǁReplicatorDynamicsǁpredict_states__mutmut_30, 
        'xǁReplicatorDynamicsǁpredict_states__mutmut_31': xǁReplicatorDynamicsǁpredict_states__mutmut_31
    }
    xǁReplicatorDynamicsǁpredict_states__mutmut_orig.__name__ = 'xǁReplicatorDynamicsǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "XXx0XX": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "X0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "XXtypeXX": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "TYPE": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "XXlistXX",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "LIST",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "XXdefaultXX": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "DEFAULT": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "XXdescriptionXX": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "DESCRIPTION": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "XXA list of initial proportions for each strategy.XX",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "a list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A LIST OF INITIAL PROPORTIONS FOR EACH STRATEGY.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "XXpayoff_matrixXX": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "PAYOFF_MATRIX": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "XXtypeXX": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "TYPE": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "XXlistXX",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "LIST",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "XXdefaultXX": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "DEFAULT": [],
                "description": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "XXdescriptionXX": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "DESCRIPTION": "The payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "XXThe payoff matrix for the game.XX",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "the payoff matrix for the game.",
            },
        }

    def xǁReplicatorDynamicsǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "THE PAYOFF MATRIX FOR THE GAME.",
            },
        }
    
    xǁReplicatorDynamicsǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_1': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_1, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_2': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_2, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_3': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_3, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_4': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_4, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_5': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_5, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_6': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_6, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_7': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_7, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_8': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_8, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_9': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_9, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_10': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_10, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_11': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_11, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_12': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_12, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_13': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_13, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_14': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_14, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_15': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_15, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_16': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_16, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_17': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_17, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_18': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_18, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_19': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_19, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_20': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_20, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_21': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_21, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_22': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_22, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_23': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_23, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_24': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_24, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_25': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_25, 
        'xǁReplicatorDynamicsǁget_parameters_schema__mutmut_26': xǁReplicatorDynamicsǁget_parameters_schema__mutmut_26
    }
    xǁReplicatorDynamicsǁget_parameters_schema__mutmut_orig.__name__ = 'xǁReplicatorDynamicsǁget_parameters_schema'
