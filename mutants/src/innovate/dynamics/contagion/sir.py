from .base import ContagionSpread
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


class SIRModel(ContagionSpread):
    """Models the spread of a contagion through a population with Susceptible,
    Infectious, and Recovered states.
    """

    def compute_spread_rate(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRModelǁcompute_spread_rate__mutmut_orig'), object.__getattribute__(self, 'xǁSIRModelǁcompute_spread_rate__mutmut_mutants'), args, kwargs, self)

    def xǁSIRModelǁcompute_spread_rate__mutmut_orig(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_1(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = None
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_2(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get(None)
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_3(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("XXSXX")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_4(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("s")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_5(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = None
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_6(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get(None)
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_7(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("XXIXX")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_8(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("i")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_9(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = None
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_10(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get(None, 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_11(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", None)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_12(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get(0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_13(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", )
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_14(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("XXtransmission_rateXX", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_15(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("TRANSMISSION_RATE", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_16(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 1.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_17(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = None

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_18(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get(None, 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_19(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", None)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_20(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get(0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_21(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", )

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_22(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("XXrecovery_rateXX", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_23(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("RECOVERY_RATE", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_24(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 1.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_25(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = None
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_26(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S / I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_27(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta / S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_28(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = +beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_29(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = None
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_30(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I + gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_31(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S / I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_32(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta / S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_33(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma / I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_34(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = None
        return dSdt, dIdt, dRdt

    def xǁSIRModelǁcompute_spread_rate__mutmut_35(self, **params):
        """Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters
        ----------
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns
        -------
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma / I
        return dSdt, dIdt, dRdt
    
    xǁSIRModelǁcompute_spread_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRModelǁcompute_spread_rate__mutmut_1': xǁSIRModelǁcompute_spread_rate__mutmut_1, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_2': xǁSIRModelǁcompute_spread_rate__mutmut_2, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_3': xǁSIRModelǁcompute_spread_rate__mutmut_3, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_4': xǁSIRModelǁcompute_spread_rate__mutmut_4, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_5': xǁSIRModelǁcompute_spread_rate__mutmut_5, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_6': xǁSIRModelǁcompute_spread_rate__mutmut_6, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_7': xǁSIRModelǁcompute_spread_rate__mutmut_7, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_8': xǁSIRModelǁcompute_spread_rate__mutmut_8, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_9': xǁSIRModelǁcompute_spread_rate__mutmut_9, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_10': xǁSIRModelǁcompute_spread_rate__mutmut_10, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_11': xǁSIRModelǁcompute_spread_rate__mutmut_11, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_12': xǁSIRModelǁcompute_spread_rate__mutmut_12, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_13': xǁSIRModelǁcompute_spread_rate__mutmut_13, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_14': xǁSIRModelǁcompute_spread_rate__mutmut_14, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_15': xǁSIRModelǁcompute_spread_rate__mutmut_15, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_16': xǁSIRModelǁcompute_spread_rate__mutmut_16, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_17': xǁSIRModelǁcompute_spread_rate__mutmut_17, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_18': xǁSIRModelǁcompute_spread_rate__mutmut_18, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_19': xǁSIRModelǁcompute_spread_rate__mutmut_19, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_20': xǁSIRModelǁcompute_spread_rate__mutmut_20, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_21': xǁSIRModelǁcompute_spread_rate__mutmut_21, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_22': xǁSIRModelǁcompute_spread_rate__mutmut_22, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_23': xǁSIRModelǁcompute_spread_rate__mutmut_23, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_24': xǁSIRModelǁcompute_spread_rate__mutmut_24, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_25': xǁSIRModelǁcompute_spread_rate__mutmut_25, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_26': xǁSIRModelǁcompute_spread_rate__mutmut_26, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_27': xǁSIRModelǁcompute_spread_rate__mutmut_27, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_28': xǁSIRModelǁcompute_spread_rate__mutmut_28, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_29': xǁSIRModelǁcompute_spread_rate__mutmut_29, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_30': xǁSIRModelǁcompute_spread_rate__mutmut_30, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_31': xǁSIRModelǁcompute_spread_rate__mutmut_31, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_32': xǁSIRModelǁcompute_spread_rate__mutmut_32, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_33': xǁSIRModelǁcompute_spread_rate__mutmut_33, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_34': xǁSIRModelǁcompute_spread_rate__mutmut_34, 
        'xǁSIRModelǁcompute_spread_rate__mutmut_35': xǁSIRModelǁcompute_spread_rate__mutmut_35
    }
    xǁSIRModelǁcompute_spread_rate__mutmut_orig.__name__ = 'xǁSIRModelǁcompute_spread_rate'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRModelǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁSIRModelǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁSIRModelǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = None
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get(None, 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", None)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get(999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", )
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("XXS0XX", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("s0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 1000)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = None
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get(None, 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", None)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get(1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", )
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("XXI0XX", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("i0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 2)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = None

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get(None, 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", None)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get(0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", )

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("XXR0XX", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("r0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=None, I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[1], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_32(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = None
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_33(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_34(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            None,
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_35(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_36(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_37(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_38(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_39(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_40(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_41(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_42(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_43(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_44(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_45(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_46(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁSIRModelǁpredict_states__mutmut_47(self, time_points, **params):
        """Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁSIRModelǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRModelǁpredict_states__mutmut_1': xǁSIRModelǁpredict_states__mutmut_1, 
        'xǁSIRModelǁpredict_states__mutmut_2': xǁSIRModelǁpredict_states__mutmut_2, 
        'xǁSIRModelǁpredict_states__mutmut_3': xǁSIRModelǁpredict_states__mutmut_3, 
        'xǁSIRModelǁpredict_states__mutmut_4': xǁSIRModelǁpredict_states__mutmut_4, 
        'xǁSIRModelǁpredict_states__mutmut_5': xǁSIRModelǁpredict_states__mutmut_5, 
        'xǁSIRModelǁpredict_states__mutmut_6': xǁSIRModelǁpredict_states__mutmut_6, 
        'xǁSIRModelǁpredict_states__mutmut_7': xǁSIRModelǁpredict_states__mutmut_7, 
        'xǁSIRModelǁpredict_states__mutmut_8': xǁSIRModelǁpredict_states__mutmut_8, 
        'xǁSIRModelǁpredict_states__mutmut_9': xǁSIRModelǁpredict_states__mutmut_9, 
        'xǁSIRModelǁpredict_states__mutmut_10': xǁSIRModelǁpredict_states__mutmut_10, 
        'xǁSIRModelǁpredict_states__mutmut_11': xǁSIRModelǁpredict_states__mutmut_11, 
        'xǁSIRModelǁpredict_states__mutmut_12': xǁSIRModelǁpredict_states__mutmut_12, 
        'xǁSIRModelǁpredict_states__mutmut_13': xǁSIRModelǁpredict_states__mutmut_13, 
        'xǁSIRModelǁpredict_states__mutmut_14': xǁSIRModelǁpredict_states__mutmut_14, 
        'xǁSIRModelǁpredict_states__mutmut_15': xǁSIRModelǁpredict_states__mutmut_15, 
        'xǁSIRModelǁpredict_states__mutmut_16': xǁSIRModelǁpredict_states__mutmut_16, 
        'xǁSIRModelǁpredict_states__mutmut_17': xǁSIRModelǁpredict_states__mutmut_17, 
        'xǁSIRModelǁpredict_states__mutmut_18': xǁSIRModelǁpredict_states__mutmut_18, 
        'xǁSIRModelǁpredict_states__mutmut_19': xǁSIRModelǁpredict_states__mutmut_19, 
        'xǁSIRModelǁpredict_states__mutmut_20': xǁSIRModelǁpredict_states__mutmut_20, 
        'xǁSIRModelǁpredict_states__mutmut_21': xǁSIRModelǁpredict_states__mutmut_21, 
        'xǁSIRModelǁpredict_states__mutmut_22': xǁSIRModelǁpredict_states__mutmut_22, 
        'xǁSIRModelǁpredict_states__mutmut_23': xǁSIRModelǁpredict_states__mutmut_23, 
        'xǁSIRModelǁpredict_states__mutmut_24': xǁSIRModelǁpredict_states__mutmut_24, 
        'xǁSIRModelǁpredict_states__mutmut_25': xǁSIRModelǁpredict_states__mutmut_25, 
        'xǁSIRModelǁpredict_states__mutmut_26': xǁSIRModelǁpredict_states__mutmut_26, 
        'xǁSIRModelǁpredict_states__mutmut_27': xǁSIRModelǁpredict_states__mutmut_27, 
        'xǁSIRModelǁpredict_states__mutmut_28': xǁSIRModelǁpredict_states__mutmut_28, 
        'xǁSIRModelǁpredict_states__mutmut_29': xǁSIRModelǁpredict_states__mutmut_29, 
        'xǁSIRModelǁpredict_states__mutmut_30': xǁSIRModelǁpredict_states__mutmut_30, 
        'xǁSIRModelǁpredict_states__mutmut_31': xǁSIRModelǁpredict_states__mutmut_31, 
        'xǁSIRModelǁpredict_states__mutmut_32': xǁSIRModelǁpredict_states__mutmut_32, 
        'xǁSIRModelǁpredict_states__mutmut_33': xǁSIRModelǁpredict_states__mutmut_33, 
        'xǁSIRModelǁpredict_states__mutmut_34': xǁSIRModelǁpredict_states__mutmut_34, 
        'xǁSIRModelǁpredict_states__mutmut_35': xǁSIRModelǁpredict_states__mutmut_35, 
        'xǁSIRModelǁpredict_states__mutmut_36': xǁSIRModelǁpredict_states__mutmut_36, 
        'xǁSIRModelǁpredict_states__mutmut_37': xǁSIRModelǁpredict_states__mutmut_37, 
        'xǁSIRModelǁpredict_states__mutmut_38': xǁSIRModelǁpredict_states__mutmut_38, 
        'xǁSIRModelǁpredict_states__mutmut_39': xǁSIRModelǁpredict_states__mutmut_39, 
        'xǁSIRModelǁpredict_states__mutmut_40': xǁSIRModelǁpredict_states__mutmut_40, 
        'xǁSIRModelǁpredict_states__mutmut_41': xǁSIRModelǁpredict_states__mutmut_41, 
        'xǁSIRModelǁpredict_states__mutmut_42': xǁSIRModelǁpredict_states__mutmut_42, 
        'xǁSIRModelǁpredict_states__mutmut_43': xǁSIRModelǁpredict_states__mutmut_43, 
        'xǁSIRModelǁpredict_states__mutmut_44': xǁSIRModelǁpredict_states__mutmut_44, 
        'xǁSIRModelǁpredict_states__mutmut_45': xǁSIRModelǁpredict_states__mutmut_45, 
        'xǁSIRModelǁpredict_states__mutmut_46': xǁSIRModelǁpredict_states__mutmut_46, 
        'xǁSIRModelǁpredict_states__mutmut_47': xǁSIRModelǁpredict_states__mutmut_47
    }
    xǁSIRModelǁpredict_states__mutmut_orig.__name__ = 'xǁSIRModelǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁSIRModelǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁSIRModelǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁSIRModelǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "XXtransmission_rateXX": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "TRANSMISSION_RATE": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "XXtypeXX": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "TYPE": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "XXfloatXX",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "FLOAT",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "XXdefaultXX": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "DEFAULT": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 1.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "XXdescriptionXX": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "DESCRIPTION": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "XXThe rate of transmission of the contagion.XX",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "the rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "THE RATE OF TRANSMISSION OF THE CONTAGION.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "XXrecovery_rateXX": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "RECOVERY_RATE": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "XXtypeXX": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "TYPE": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "XXfloatXX",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "FLOAT",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "XXdefaultXX": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "DEFAULT": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 1.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "XXdescriptionXX": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "DESCRIPTION": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "XXThe rate of recovery from the contagion.XX",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "the rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "THE RATE OF RECOVERY FROM THE CONTAGION.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_29(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "XXS0XX": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_30(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "s0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_31(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "XXtypeXX": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_32(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "TYPE": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_33(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "XXfloatXX",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_34(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "FLOAT",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_35(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "XXdefaultXX": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_36(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "DEFAULT": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_37(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 1000,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_38(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "XXdescriptionXX": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_39(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "DESCRIPTION": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_40(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "XXThe initial number of susceptible individuals.XX",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_41(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "the initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_42(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "THE INITIAL NUMBER OF SUSCEPTIBLE INDIVIDUALS.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_43(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "XXI0XX": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_44(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "i0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_45(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "XXtypeXX": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_46(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "TYPE": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_47(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "XXfloatXX",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_48(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "FLOAT",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_49(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "XXdefaultXX": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_50(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "DEFAULT": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_51(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 2,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_52(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "XXdescriptionXX": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_53(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "DESCRIPTION": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_54(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "XXThe initial number of infectious individuals.XX",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_55(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "the initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_56(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "THE INITIAL NUMBER OF INFECTIOUS INDIVIDUALS.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_57(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "XXR0XX": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_58(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "r0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_59(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "XXtypeXX": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_60(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "TYPE": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_61(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "XXfloatXX",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_62(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "FLOAT",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_63(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "XXdefaultXX": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_64(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "DEFAULT": 0,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_65(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_66(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "XXdescriptionXX": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_67(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "DESCRIPTION": "The initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_68(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "XXThe initial number of recovered individuals.XX",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_69(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "the initial number of recovered individuals.",
            },
        }

    def xǁSIRModelǁget_parameters_schema__mutmut_70(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "THE INITIAL NUMBER OF RECOVERED INDIVIDUALS.",
            },
        }
    
    xǁSIRModelǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁSIRModelǁget_parameters_schema__mutmut_1': xǁSIRModelǁget_parameters_schema__mutmut_1, 
        'xǁSIRModelǁget_parameters_schema__mutmut_2': xǁSIRModelǁget_parameters_schema__mutmut_2, 
        'xǁSIRModelǁget_parameters_schema__mutmut_3': xǁSIRModelǁget_parameters_schema__mutmut_3, 
        'xǁSIRModelǁget_parameters_schema__mutmut_4': xǁSIRModelǁget_parameters_schema__mutmut_4, 
        'xǁSIRModelǁget_parameters_schema__mutmut_5': xǁSIRModelǁget_parameters_schema__mutmut_5, 
        'xǁSIRModelǁget_parameters_schema__mutmut_6': xǁSIRModelǁget_parameters_schema__mutmut_6, 
        'xǁSIRModelǁget_parameters_schema__mutmut_7': xǁSIRModelǁget_parameters_schema__mutmut_7, 
        'xǁSIRModelǁget_parameters_schema__mutmut_8': xǁSIRModelǁget_parameters_schema__mutmut_8, 
        'xǁSIRModelǁget_parameters_schema__mutmut_9': xǁSIRModelǁget_parameters_schema__mutmut_9, 
        'xǁSIRModelǁget_parameters_schema__mutmut_10': xǁSIRModelǁget_parameters_schema__mutmut_10, 
        'xǁSIRModelǁget_parameters_schema__mutmut_11': xǁSIRModelǁget_parameters_schema__mutmut_11, 
        'xǁSIRModelǁget_parameters_schema__mutmut_12': xǁSIRModelǁget_parameters_schema__mutmut_12, 
        'xǁSIRModelǁget_parameters_schema__mutmut_13': xǁSIRModelǁget_parameters_schema__mutmut_13, 
        'xǁSIRModelǁget_parameters_schema__mutmut_14': xǁSIRModelǁget_parameters_schema__mutmut_14, 
        'xǁSIRModelǁget_parameters_schema__mutmut_15': xǁSIRModelǁget_parameters_schema__mutmut_15, 
        'xǁSIRModelǁget_parameters_schema__mutmut_16': xǁSIRModelǁget_parameters_schema__mutmut_16, 
        'xǁSIRModelǁget_parameters_schema__mutmut_17': xǁSIRModelǁget_parameters_schema__mutmut_17, 
        'xǁSIRModelǁget_parameters_schema__mutmut_18': xǁSIRModelǁget_parameters_schema__mutmut_18, 
        'xǁSIRModelǁget_parameters_schema__mutmut_19': xǁSIRModelǁget_parameters_schema__mutmut_19, 
        'xǁSIRModelǁget_parameters_schema__mutmut_20': xǁSIRModelǁget_parameters_schema__mutmut_20, 
        'xǁSIRModelǁget_parameters_schema__mutmut_21': xǁSIRModelǁget_parameters_schema__mutmut_21, 
        'xǁSIRModelǁget_parameters_schema__mutmut_22': xǁSIRModelǁget_parameters_schema__mutmut_22, 
        'xǁSIRModelǁget_parameters_schema__mutmut_23': xǁSIRModelǁget_parameters_schema__mutmut_23, 
        'xǁSIRModelǁget_parameters_schema__mutmut_24': xǁSIRModelǁget_parameters_schema__mutmut_24, 
        'xǁSIRModelǁget_parameters_schema__mutmut_25': xǁSIRModelǁget_parameters_schema__mutmut_25, 
        'xǁSIRModelǁget_parameters_schema__mutmut_26': xǁSIRModelǁget_parameters_schema__mutmut_26, 
        'xǁSIRModelǁget_parameters_schema__mutmut_27': xǁSIRModelǁget_parameters_schema__mutmut_27, 
        'xǁSIRModelǁget_parameters_schema__mutmut_28': xǁSIRModelǁget_parameters_schema__mutmut_28, 
        'xǁSIRModelǁget_parameters_schema__mutmut_29': xǁSIRModelǁget_parameters_schema__mutmut_29, 
        'xǁSIRModelǁget_parameters_schema__mutmut_30': xǁSIRModelǁget_parameters_schema__mutmut_30, 
        'xǁSIRModelǁget_parameters_schema__mutmut_31': xǁSIRModelǁget_parameters_schema__mutmut_31, 
        'xǁSIRModelǁget_parameters_schema__mutmut_32': xǁSIRModelǁget_parameters_schema__mutmut_32, 
        'xǁSIRModelǁget_parameters_schema__mutmut_33': xǁSIRModelǁget_parameters_schema__mutmut_33, 
        'xǁSIRModelǁget_parameters_schema__mutmut_34': xǁSIRModelǁget_parameters_schema__mutmut_34, 
        'xǁSIRModelǁget_parameters_schema__mutmut_35': xǁSIRModelǁget_parameters_schema__mutmut_35, 
        'xǁSIRModelǁget_parameters_schema__mutmut_36': xǁSIRModelǁget_parameters_schema__mutmut_36, 
        'xǁSIRModelǁget_parameters_schema__mutmut_37': xǁSIRModelǁget_parameters_schema__mutmut_37, 
        'xǁSIRModelǁget_parameters_schema__mutmut_38': xǁSIRModelǁget_parameters_schema__mutmut_38, 
        'xǁSIRModelǁget_parameters_schema__mutmut_39': xǁSIRModelǁget_parameters_schema__mutmut_39, 
        'xǁSIRModelǁget_parameters_schema__mutmut_40': xǁSIRModelǁget_parameters_schema__mutmut_40, 
        'xǁSIRModelǁget_parameters_schema__mutmut_41': xǁSIRModelǁget_parameters_schema__mutmut_41, 
        'xǁSIRModelǁget_parameters_schema__mutmut_42': xǁSIRModelǁget_parameters_schema__mutmut_42, 
        'xǁSIRModelǁget_parameters_schema__mutmut_43': xǁSIRModelǁget_parameters_schema__mutmut_43, 
        'xǁSIRModelǁget_parameters_schema__mutmut_44': xǁSIRModelǁget_parameters_schema__mutmut_44, 
        'xǁSIRModelǁget_parameters_schema__mutmut_45': xǁSIRModelǁget_parameters_schema__mutmut_45, 
        'xǁSIRModelǁget_parameters_schema__mutmut_46': xǁSIRModelǁget_parameters_schema__mutmut_46, 
        'xǁSIRModelǁget_parameters_schema__mutmut_47': xǁSIRModelǁget_parameters_schema__mutmut_47, 
        'xǁSIRModelǁget_parameters_schema__mutmut_48': xǁSIRModelǁget_parameters_schema__mutmut_48, 
        'xǁSIRModelǁget_parameters_schema__mutmut_49': xǁSIRModelǁget_parameters_schema__mutmut_49, 
        'xǁSIRModelǁget_parameters_schema__mutmut_50': xǁSIRModelǁget_parameters_schema__mutmut_50, 
        'xǁSIRModelǁget_parameters_schema__mutmut_51': xǁSIRModelǁget_parameters_schema__mutmut_51, 
        'xǁSIRModelǁget_parameters_schema__mutmut_52': xǁSIRModelǁget_parameters_schema__mutmut_52, 
        'xǁSIRModelǁget_parameters_schema__mutmut_53': xǁSIRModelǁget_parameters_schema__mutmut_53, 
        'xǁSIRModelǁget_parameters_schema__mutmut_54': xǁSIRModelǁget_parameters_schema__mutmut_54, 
        'xǁSIRModelǁget_parameters_schema__mutmut_55': xǁSIRModelǁget_parameters_schema__mutmut_55, 
        'xǁSIRModelǁget_parameters_schema__mutmut_56': xǁSIRModelǁget_parameters_schema__mutmut_56, 
        'xǁSIRModelǁget_parameters_schema__mutmut_57': xǁSIRModelǁget_parameters_schema__mutmut_57, 
        'xǁSIRModelǁget_parameters_schema__mutmut_58': xǁSIRModelǁget_parameters_schema__mutmut_58, 
        'xǁSIRModelǁget_parameters_schema__mutmut_59': xǁSIRModelǁget_parameters_schema__mutmut_59, 
        'xǁSIRModelǁget_parameters_schema__mutmut_60': xǁSIRModelǁget_parameters_schema__mutmut_60, 
        'xǁSIRModelǁget_parameters_schema__mutmut_61': xǁSIRModelǁget_parameters_schema__mutmut_61, 
        'xǁSIRModelǁget_parameters_schema__mutmut_62': xǁSIRModelǁget_parameters_schema__mutmut_62, 
        'xǁSIRModelǁget_parameters_schema__mutmut_63': xǁSIRModelǁget_parameters_schema__mutmut_63, 
        'xǁSIRModelǁget_parameters_schema__mutmut_64': xǁSIRModelǁget_parameters_schema__mutmut_64, 
        'xǁSIRModelǁget_parameters_schema__mutmut_65': xǁSIRModelǁget_parameters_schema__mutmut_65, 
        'xǁSIRModelǁget_parameters_schema__mutmut_66': xǁSIRModelǁget_parameters_schema__mutmut_66, 
        'xǁSIRModelǁget_parameters_schema__mutmut_67': xǁSIRModelǁget_parameters_schema__mutmut_67, 
        'xǁSIRModelǁget_parameters_schema__mutmut_68': xǁSIRModelǁget_parameters_schema__mutmut_68, 
        'xǁSIRModelǁget_parameters_schema__mutmut_69': xǁSIRModelǁget_parameters_schema__mutmut_69, 
        'xǁSIRModelǁget_parameters_schema__mutmut_70': xǁSIRModelǁget_parameters_schema__mutmut_70
    }
    xǁSIRModelǁget_parameters_schema__mutmut_orig.__name__ = 'xǁSIRModelǁget_parameters_schema'
