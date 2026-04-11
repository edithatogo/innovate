from .base import SystemBehavior
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


class HypeCycleBehavior(SystemBehavior):
    """Models the rise and fall of expectations through coupled expectation and
    maturity stocks.
    """

    def compute_behavior_rates(self, **params):
        args = []# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_orig'), object.__getattribute__(self, 'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_mutants'), args, kwargs, self)

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_orig(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_1(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = None
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_2(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get(None)
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_3(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("XXEXX")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_4(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("e")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_5(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = None

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_6(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get(None)

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_7(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("XXMXX")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_8(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("m")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_9(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = None
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_10(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get(None, 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_11(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", None)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_12(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get(0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_13(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", )
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_14(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("XXalpha1XX", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_15(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("ALPHA1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_16(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 1.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_17(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = None
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_18(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get(None, 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_19(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", None)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_20(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get(0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_21(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", )
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_22(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("XXalpha2XX", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_23(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("ALPHA2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_24(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 1.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_25(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = None
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_26(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get(None, 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_27(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", None)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_28(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get(0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_29(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", )
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_30(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("XXalpha3XX", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_31(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("ALPHA3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_32(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 1.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_33(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = None
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_34(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get(None, 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_35(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", None)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_36(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get(0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_37(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", )
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_38(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("XXalpha4XX", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_39(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("ALPHA4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_40(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 1.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_41(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = None

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_42(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get(None, 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_43(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", None)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_44(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get(0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_45(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", )

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_46(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("XXinnovation_triggerXX", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_47(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("INNOVATION_TRIGGER", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_48(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 1)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_49(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = None
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_50(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get(None, 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_51(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", None)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_52(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get(0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_53(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", )
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_54(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("XXbeta1XX", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_55(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("BETA1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_56(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 1.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_57(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = None
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_58(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get(None, 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_59(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", None)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_60(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get(0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_61(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", )
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_62(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("XXbeta2XX", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_63(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("BETA2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_64(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 1.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_65(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = None

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_66(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get(None, 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_67(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", None)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_68(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get(0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_69(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", )

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_70(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("XXbeta3XX", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_71(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("BETA3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_72(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 1.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_73(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = None

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_74(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) / E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_75(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get(None, 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_76(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", None) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_77(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get(0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_78(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", ) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_79(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("XXrd_investment_factorXX", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_80(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("RD_INVESTMENT_FACTOR", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_81(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 1.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_82(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = None
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_83(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E - alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_84(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E + alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_85(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger - alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_86(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 / innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_87(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M / E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_88(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 / M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_89(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 / E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_90(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) / E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_91(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 / (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_92(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E + M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_93(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = None

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_94(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M + beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_95(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment - beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_96(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 / rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_97(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 / M - beta3 * M

        return dEdt, dMdt

    def xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_98(self, **params):
        """Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.

        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.

        Returns
        -------
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 / M

        return dEdt, dMdt
    
    xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_1': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_1, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_2': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_2, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_3': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_3, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_4': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_4, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_5': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_5, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_6': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_6, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_7': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_7, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_8': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_8, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_9': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_9, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_10': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_10, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_11': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_11, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_12': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_12, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_13': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_13, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_14': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_14, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_15': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_15, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_16': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_16, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_17': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_17, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_18': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_18, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_19': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_19, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_20': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_20, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_21': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_21, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_22': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_22, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_23': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_23, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_24': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_24, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_25': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_25, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_26': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_26, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_27': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_27, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_28': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_28, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_29': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_29, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_30': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_30, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_31': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_31, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_32': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_32, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_33': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_33, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_34': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_34, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_35': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_35, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_36': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_36, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_37': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_37, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_38': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_38, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_39': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_39, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_40': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_40, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_41': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_41, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_42': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_42, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_43': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_43, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_44': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_44, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_45': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_45, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_46': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_46, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_47': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_47, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_48': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_48, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_49': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_49, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_50': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_50, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_51': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_51, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_52': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_52, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_53': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_53, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_54': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_54, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_55': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_55, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_56': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_56, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_57': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_57, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_58': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_58, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_59': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_59, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_60': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_60, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_61': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_61, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_62': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_62, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_63': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_63, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_64': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_64, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_65': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_65, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_66': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_66, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_67': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_67, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_68': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_68, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_69': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_69, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_70': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_70, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_71': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_71, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_72': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_72, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_73': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_73, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_74': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_74, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_75': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_75, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_76': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_76, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_77': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_77, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_78': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_78, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_79': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_79, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_80': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_80, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_81': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_81, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_82': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_82, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_83': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_83, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_84': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_84, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_85': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_85, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_86': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_86, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_87': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_87, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_88': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_88, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_89': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_89, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_90': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_90, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_91': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_91, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_92': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_92, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_93': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_93, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_94': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_94, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_95': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_95, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_96': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_96, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_97': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_97, 
        'xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_98': xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_98
    }
    xǁHypeCycleBehaviorǁcompute_behavior_rates__mutmut_orig.__name__ = 'xǁHypeCycleBehaviorǁcompute_behavior_rates'

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeCycleBehaviorǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁHypeCycleBehaviorǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = None
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get(None, 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", None)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get(1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", )
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("XXE0XX", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("e0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 2)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = None

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get(None, 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", None)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get(1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", )

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("XXM0XX", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("m0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 2)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=None, M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=None, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_21(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_22(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[1], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_23(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_24(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = None
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_25(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_26(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            None,
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_27(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_28(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_29(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method=None,
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_30(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_31(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_32(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_33(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_34(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_35(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_36(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_37(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [E0, M0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_38(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁHypeCycleBehaviorǁpredict_states__mutmut_39(self, time_points, **params):
        """Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to evaluate the system states.

        Returns
        -------
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        def ode_func(t, y):
            return self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.T
    
    xǁHypeCycleBehaviorǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeCycleBehaviorǁpredict_states__mutmut_1': xǁHypeCycleBehaviorǁpredict_states__mutmut_1, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_2': xǁHypeCycleBehaviorǁpredict_states__mutmut_2, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_3': xǁHypeCycleBehaviorǁpredict_states__mutmut_3, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_4': xǁHypeCycleBehaviorǁpredict_states__mutmut_4, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_5': xǁHypeCycleBehaviorǁpredict_states__mutmut_5, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_6': xǁHypeCycleBehaviorǁpredict_states__mutmut_6, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_7': xǁHypeCycleBehaviorǁpredict_states__mutmut_7, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_8': xǁHypeCycleBehaviorǁpredict_states__mutmut_8, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_9': xǁHypeCycleBehaviorǁpredict_states__mutmut_9, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_10': xǁHypeCycleBehaviorǁpredict_states__mutmut_10, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_11': xǁHypeCycleBehaviorǁpredict_states__mutmut_11, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_12': xǁHypeCycleBehaviorǁpredict_states__mutmut_12, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_13': xǁHypeCycleBehaviorǁpredict_states__mutmut_13, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_14': xǁHypeCycleBehaviorǁpredict_states__mutmut_14, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_15': xǁHypeCycleBehaviorǁpredict_states__mutmut_15, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_16': xǁHypeCycleBehaviorǁpredict_states__mutmut_16, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_17': xǁHypeCycleBehaviorǁpredict_states__mutmut_17, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_18': xǁHypeCycleBehaviorǁpredict_states__mutmut_18, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_19': xǁHypeCycleBehaviorǁpredict_states__mutmut_19, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_20': xǁHypeCycleBehaviorǁpredict_states__mutmut_20, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_21': xǁHypeCycleBehaviorǁpredict_states__mutmut_21, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_22': xǁHypeCycleBehaviorǁpredict_states__mutmut_22, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_23': xǁHypeCycleBehaviorǁpredict_states__mutmut_23, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_24': xǁHypeCycleBehaviorǁpredict_states__mutmut_24, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_25': xǁHypeCycleBehaviorǁpredict_states__mutmut_25, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_26': xǁHypeCycleBehaviorǁpredict_states__mutmut_26, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_27': xǁHypeCycleBehaviorǁpredict_states__mutmut_27, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_28': xǁHypeCycleBehaviorǁpredict_states__mutmut_28, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_29': xǁHypeCycleBehaviorǁpredict_states__mutmut_29, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_30': xǁHypeCycleBehaviorǁpredict_states__mutmut_30, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_31': xǁHypeCycleBehaviorǁpredict_states__mutmut_31, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_32': xǁHypeCycleBehaviorǁpredict_states__mutmut_32, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_33': xǁHypeCycleBehaviorǁpredict_states__mutmut_33, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_34': xǁHypeCycleBehaviorǁpredict_states__mutmut_34, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_35': xǁHypeCycleBehaviorǁpredict_states__mutmut_35, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_36': xǁHypeCycleBehaviorǁpredict_states__mutmut_36, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_37': xǁHypeCycleBehaviorǁpredict_states__mutmut_37, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_38': xǁHypeCycleBehaviorǁpredict_states__mutmut_38, 
        'xǁHypeCycleBehaviorǁpredict_states__mutmut_39': xǁHypeCycleBehaviorǁpredict_states__mutmut_39
    }
    xǁHypeCycleBehaviorǁpredict_states__mutmut_orig.__name__ = 'xǁHypeCycleBehaviorǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "XXalpha1XX": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "ALPHA1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"XXtypeXX": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"TYPE": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "XXfloatXX", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "FLOAT", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "XXdefaultXX": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "DEFAULT": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 1.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "XXalpha2XX": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "ALPHA2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"XXtypeXX": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"TYPE": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "XXfloatXX", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "FLOAT", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "XXdefaultXX": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "DEFAULT": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 1.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "XXalpha3XX": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "ALPHA3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"XXtypeXX": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"TYPE": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "XXfloatXX", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "FLOAT", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "XXdefaultXX": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "DEFAULT": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 1.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "XXalpha4XX": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_29(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "ALPHA4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_30(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"XXtypeXX": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_31(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"TYPE": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_32(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "XXfloatXX", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_33(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "FLOAT", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_34(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "XXdefaultXX": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_35(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "DEFAULT": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_36(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 1.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_37(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "XXinnovation_triggerXX": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_38(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "INNOVATION_TRIGGER": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_39(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"XXtypeXX": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_40(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"TYPE": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_41(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "XXfloatXX", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_42(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "FLOAT", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_43(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "XXdefaultXX": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_44(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "DEFAULT": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_45(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 1},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_46(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "XXbeta1XX": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_47(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "BETA1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_48(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"XXtypeXX": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_49(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"TYPE": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_50(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "XXfloatXX", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_51(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "FLOAT", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_52(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "XXdefaultXX": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_53(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "DEFAULT": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_54(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 1.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_55(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "XXbeta2XX": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_56(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "BETA2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_57(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"XXtypeXX": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_58(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"TYPE": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_59(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "XXfloatXX", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_60(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "FLOAT", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_61(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "XXdefaultXX": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_62(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "DEFAULT": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_63(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 1.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_64(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "XXbeta3XX": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_65(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "BETA3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_66(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"XXtypeXX": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_67(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"TYPE": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_68(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "XXfloatXX", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_69(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "FLOAT", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_70(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "XXdefaultXX": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_71(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "DEFAULT": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_72(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 1.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_73(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "XXrd_investment_factorXX": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_74(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "RD_INVESTMENT_FACTOR": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_75(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"XXtypeXX": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_76(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"TYPE": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_77(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "XXfloatXX", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_78(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "FLOAT", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_79(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "XXdefaultXX": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_80(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "DEFAULT": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_81(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 1.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_82(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "XXE0XX": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_83(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "e0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_84(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"XXtypeXX": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_85(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"TYPE": "float", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_86(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "XXfloatXX", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_87(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "FLOAT", "default": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_88(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "XXdefaultXX": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_89(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "DEFAULT": 1},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_90(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 2},
            "M0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_91(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "XXM0XX": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_92(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "m0": {"type": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_93(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"XXtypeXX": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_94(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"TYPE": "float", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_95(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "XXfloatXX", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_96(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "FLOAT", "default": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_97(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "XXdefaultXX": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_98(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "DEFAULT": 1},
        }

    def xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_99(self):
        """Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.

        Returns
        -------
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 2},
        }
    
    xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_1': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_1, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_2': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_2, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_3': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_3, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_4': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_4, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_5': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_5, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_6': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_6, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_7': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_7, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_8': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_8, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_9': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_9, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_10': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_10, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_11': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_11, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_12': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_12, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_13': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_13, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_14': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_14, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_15': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_15, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_16': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_16, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_17': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_17, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_18': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_18, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_19': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_19, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_20': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_20, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_21': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_21, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_22': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_22, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_23': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_23, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_24': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_24, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_25': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_25, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_26': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_26, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_27': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_27, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_28': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_28, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_29': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_29, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_30': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_30, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_31': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_31, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_32': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_32, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_33': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_33, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_34': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_34, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_35': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_35, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_36': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_36, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_37': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_37, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_38': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_38, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_39': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_39, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_40': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_40, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_41': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_41, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_42': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_42, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_43': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_43, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_44': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_44, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_45': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_45, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_46': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_46, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_47': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_47, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_48': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_48, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_49': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_49, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_50': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_50, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_51': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_51, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_52': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_52, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_53': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_53, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_54': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_54, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_55': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_55, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_56': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_56, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_57': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_57, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_58': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_58, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_59': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_59, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_60': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_60, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_61': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_61, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_62': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_62, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_63': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_63, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_64': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_64, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_65': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_65, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_66': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_66, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_67': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_67, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_68': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_68, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_69': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_69, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_70': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_70, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_71': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_71, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_72': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_72, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_73': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_73, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_74': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_74, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_75': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_75, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_76': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_76, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_77': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_77, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_78': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_78, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_79': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_79, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_80': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_80, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_81': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_81, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_82': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_82, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_83': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_83, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_84': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_84, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_85': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_85, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_86': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_86, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_87': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_87, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_88': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_88, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_89': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_89, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_90': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_90, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_91': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_91, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_92': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_92, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_93': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_93, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_94': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_94, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_95': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_95, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_96': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_96, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_97': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_97, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_98': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_98, 
        'xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_99': xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_99
    }
    xǁHypeCycleBehaviorǁget_parameters_schema__mutmut_orig.__name__ = 'xǁHypeCycleBehaviorǁget_parameters_schema'
