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


class LotkaVolterraCompetition(CompetitiveInteraction):
    """Models the competition between two species using the Lotka-Volterra equations."""

    def compute_interaction_rates(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_orig'), object.__getattribute__(self, 'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_mutants'), args, kwargs, self)

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_orig(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_1(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = None
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_2(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get(None)
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_3(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("XXN1XX")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_4(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("n1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_5(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = None
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_6(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get(None)
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_7(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("XXN2XX")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_8(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("n2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_9(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = None
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_10(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get(None, 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_11(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", None)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_12(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get(0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_13(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", )
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_14(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("XXgrowth_rate_1XX", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_15(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("GROWTH_RATE_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_16(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 1.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_17(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = None
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_18(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get(None, 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_19(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", None)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_20(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get(0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_21(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", )
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_22(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("XXgrowth_rate_2XX", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_23(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("GROWTH_RATE_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_24(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 1.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_25(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = None
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_26(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get(None, 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_27(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", None)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_28(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get(1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_29(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", )
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_30(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("XXcarrying_capacity_1XX", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_31(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("CARRYING_CAPACITY_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_32(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1001)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_33(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = None
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_34(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get(None, 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_35(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", None)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_36(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get(1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_37(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", )
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_38(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("XXcarrying_capacity_2XX", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_39(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("CARRYING_CAPACITY_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_40(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1001)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_41(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = None
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_42(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get(None, 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_43(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", None)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_44(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get(1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_45(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", )
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_46(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("XXcompetition_coeff_12XX", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_47(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("COMPETITION_COEFF_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_48(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 2.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_49(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = None

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_50(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get(None, 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_51(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", None)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_52(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get(1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_53(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", )

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_54(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("XXcompetition_coeff_21XX", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_55(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("COMPETITION_COEFF_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_56(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 2.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_57(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = None
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_58(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 / (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_59(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 / N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_60(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 + (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_61(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (2 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_62(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) * K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_63(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 - alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_64(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 / N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_65(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = None
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_66(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 / (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_67(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 / N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_68(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 + (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_69(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (2 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_70(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) * K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_71(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 - alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_72(self, **params):
        """Calculates the instantaneous interaction rates.

        Equations:
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.

        Parameters
        ----------
                N1 (float): Current population of species 1.
                N2 (float): Current population of species 2.

        Returns
        -------
                tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 / N1) / K2)
        return dN1dt, dN2dt
    
    xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_1': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_1, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_2': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_2, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_3': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_3, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_4': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_4, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_5': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_5, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_6': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_6, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_7': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_7, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_8': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_8, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_9': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_9, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_10': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_10, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_11': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_11, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_12': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_12, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_13': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_13, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_14': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_14, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_15': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_15, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_16': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_16, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_17': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_17, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_18': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_18, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_19': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_19, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_20': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_20, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_21': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_21, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_22': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_22, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_23': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_23, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_24': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_24, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_25': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_25, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_26': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_26, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_27': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_27, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_28': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_28, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_29': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_29, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_30': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_30, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_31': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_31, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_32': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_32, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_33': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_33, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_34': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_34, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_35': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_35, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_36': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_36, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_37': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_37, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_38': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_38, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_39': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_39, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_40': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_40, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_41': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_41, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_42': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_42, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_43': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_43, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_44': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_44, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_45': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_45, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_46': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_46, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_47': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_47, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_48': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_48, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_49': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_49, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_50': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_50, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_51': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_51, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_52': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_52, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_53': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_53, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_54': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_54, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_55': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_55, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_56': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_56, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_57': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_57, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_58': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_58, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_59': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_59, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_60': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_60, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_61': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_61, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_62': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_62, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_63': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_63, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_64': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_64, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_65': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_65, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_66': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_66, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_67': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_67, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_68': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_68, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_69': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_69, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_70': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_70, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_71': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_71, 
        'xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_72': xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_72
    }
    xǁLotkaVolterraCompetitionǁcompute_interaction_rates__mutmut_orig.__name__ = 'xǁLotkaVolterraCompetitionǁcompute_interaction_rates'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = None
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get(None, 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", None)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get(1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", )
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("XXN1_0XX", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("n1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 2)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = None

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get(None, 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", None)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get(1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", )

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("XXN2_0XX", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("n2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 2)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=None, N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[1], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = None
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            None,
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_32(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_33(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_34(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_35(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_36(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_37(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_38(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁLotkaVolterraCompetitionǁpredict_states__mutmut_39(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the populations.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        def ode_func(t, y):
            return self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁLotkaVolterraCompetitionǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_1': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_1, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_2': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_2, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_3': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_3, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_4': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_4, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_5': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_5, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_6': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_6, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_7': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_7, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_8': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_8, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_9': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_9, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_10': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_10, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_11': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_11, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_12': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_12, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_13': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_13, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_14': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_14, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_15': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_15, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_16': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_16, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_17': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_17, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_18': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_18, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_19': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_19, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_20': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_20, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_21': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_21, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_22': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_22, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_23': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_23, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_24': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_24, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_25': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_25, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_26': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_26, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_27': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_27, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_28': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_28, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_29': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_29, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_30': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_30, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_31': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_31, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_32': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_32, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_33': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_33, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_34': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_34, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_35': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_35, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_36': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_36, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_37': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_37, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_38': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_38, 
        'xǁLotkaVolterraCompetitionǁpredict_states__mutmut_39': xǁLotkaVolterraCompetitionǁpredict_states__mutmut_39
    }
    xǁLotkaVolterraCompetitionǁpredict_states__mutmut_orig.__name__ = 'xǁLotkaVolterraCompetitionǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "XXgrowth_rate_1XX": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "GROWTH_RATE_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "XXtypeXX": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "TYPE": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "XXfloatXX",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "FLOAT",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "XXdefaultXX": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "DEFAULT": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 1.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "XXdescriptionXX": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "DESCRIPTION": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "XXThe intrinsic growth rate of species 1.XX",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "the intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "THE INTRINSIC GROWTH RATE OF SPECIES 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "XXgrowth_rate_2XX": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "GROWTH_RATE_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "XXtypeXX": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "TYPE": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "XXfloatXX",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "FLOAT",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "XXdefaultXX": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "DEFAULT": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 1.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "XXdescriptionXX": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "DESCRIPTION": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "XXThe intrinsic growth rate of species 2.XX",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "the intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "THE INTRINSIC GROWTH RATE OF SPECIES 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_29(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "XXcarrying_capacity_1XX": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_30(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "CARRYING_CAPACITY_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_31(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "XXtypeXX": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_32(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "TYPE": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_33(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "XXfloatXX",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_34(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "FLOAT",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_35(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "XXdefaultXX": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_36(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "DEFAULT": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_37(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1001,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_38(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "XXdescriptionXX": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_39(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "DESCRIPTION": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_40(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "XXThe carrying capacity of species 1.XX",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_41(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "the carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_42(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "THE CARRYING CAPACITY OF SPECIES 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_43(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "XXcarrying_capacity_2XX": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_44(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "CARRYING_CAPACITY_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_45(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "XXtypeXX": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_46(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "TYPE": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_47(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "XXfloatXX",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_48(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "FLOAT",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_49(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "XXdefaultXX": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_50(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "DEFAULT": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_51(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1001,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_52(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "XXdescriptionXX": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_53(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "DESCRIPTION": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_54(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "XXThe carrying capacity of species 2.XX",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_55(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "the carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_56(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "THE CARRYING CAPACITY OF SPECIES 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_57(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "XXcompetition_coeff_12XX": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_58(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "COMPETITION_COEFF_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_59(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "XXtypeXX": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_60(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "TYPE": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_61(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "XXfloatXX",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_62(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "FLOAT",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_63(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "XXdefaultXX": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_64(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "DEFAULT": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_65(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 2.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_66(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "XXdescriptionXX": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_67(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "DESCRIPTION": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_68(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "XXThe competition coefficient of species 2 on species 1.XX",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_69(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "the competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_70(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "THE COMPETITION COEFFICIENT OF SPECIES 2 ON SPECIES 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_71(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "XXcompetition_coeff_21XX": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_72(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "COMPETITION_COEFF_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_73(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "XXtypeXX": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_74(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "TYPE": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_75(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "XXfloatXX",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_76(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "FLOAT",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_77(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "XXdefaultXX": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_78(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "DEFAULT": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_79(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 2.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_80(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "XXdescriptionXX": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_81(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "DESCRIPTION": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_82(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "XXThe competition coefficient of species 1 on species 2.XX",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_83(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "the competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_84(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "THE COMPETITION COEFFICIENT OF SPECIES 1 ON SPECIES 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_85(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "XXN1_0XX": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_86(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "n1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_87(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "XXtypeXX": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_88(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "TYPE": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_89(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "XXfloatXX",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_90(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "FLOAT",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_91(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "XXdefaultXX": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_92(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "DEFAULT": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_93(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 2,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_94(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "XXdescriptionXX": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_95(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "DESCRIPTION": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_96(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "XXThe initial population of species 1.XX",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_97(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "the initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_98(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "THE INITIAL POPULATION OF SPECIES 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_99(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "XXN2_0XX": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_100(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "n2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_101(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "XXtypeXX": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_102(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "TYPE": "float",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_103(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "XXfloatXX",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_104(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "FLOAT",
                "default": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_105(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "XXdefaultXX": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_106(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "DEFAULT": 1,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_107(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 2,
                "description": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_108(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "XXdescriptionXX": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_109(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "DESCRIPTION": "The initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_110(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "XXThe initial population of species 2.XX",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_111(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "the initial population of species 2.",
            },
        }

    def xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_112(self):
        """Returns the schema for the model's parameters.

        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1.",
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2.",
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1.",
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2.",
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1.",
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2.",
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1.",
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "THE INITIAL POPULATION OF SPECIES 2.",
            },
        }
    
    xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_1': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_1, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_2': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_2, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_3': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_3, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_4': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_4, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_5': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_5, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_6': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_6, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_7': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_7, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_8': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_8, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_9': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_9, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_10': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_10, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_11': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_11, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_12': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_12, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_13': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_13, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_14': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_14, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_15': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_15, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_16': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_16, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_17': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_17, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_18': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_18, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_19': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_19, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_20': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_20, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_21': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_21, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_22': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_22, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_23': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_23, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_24': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_24, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_25': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_25, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_26': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_26, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_27': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_27, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_28': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_28, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_29': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_29, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_30': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_30, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_31': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_31, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_32': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_32, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_33': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_33, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_34': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_34, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_35': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_35, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_36': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_36, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_37': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_37, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_38': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_38, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_39': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_39, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_40': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_40, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_41': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_41, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_42': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_42, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_43': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_43, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_44': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_44, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_45': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_45, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_46': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_46, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_47': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_47, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_48': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_48, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_49': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_49, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_50': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_50, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_51': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_51, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_52': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_52, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_53': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_53, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_54': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_54, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_55': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_55, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_56': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_56, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_57': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_57, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_58': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_58, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_59': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_59, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_60': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_60, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_61': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_61, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_62': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_62, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_63': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_63, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_64': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_64, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_65': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_65, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_66': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_66, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_67': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_67, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_68': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_68, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_69': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_69, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_70': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_70, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_71': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_71, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_72': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_72, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_73': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_73, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_74': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_74, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_75': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_75, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_76': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_76, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_77': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_77, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_78': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_78, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_79': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_79, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_80': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_80, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_81': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_81, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_82': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_82, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_83': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_83, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_84': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_84, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_85': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_85, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_86': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_86, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_87': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_87, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_88': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_88, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_89': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_89, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_90': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_90, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_91': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_91, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_92': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_92, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_93': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_93, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_94': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_94, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_95': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_95, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_96': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_96, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_97': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_97, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_98': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_98, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_99': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_99, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_100': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_100, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_101': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_101, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_102': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_102, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_103': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_103, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_104': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_104, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_105': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_105, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_106': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_106, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_107': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_107, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_108': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_108, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_109': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_109, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_110': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_110, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_111': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_111, 
        'xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_112': xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_112
    }
    xǁLotkaVolterraCompetitionǁget_parameters_schema__mutmut_orig.__name__ = 'xǁLotkaVolterraCompetitionǁget_parameters_schema'
