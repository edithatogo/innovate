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


class DualInfluenceGrowth(GrowthCurve):
    """Models growth driven by two distinct forces: external influence (innovators)
    and internal influence (imitators). The shape of the S-curve can be symmetric
    or asymmetric depending on the relative strength of these two forces. This is
    often referred to as the Bass model.

    Core Behavior: Captures the dynamics of products or ideas that are initially
    pushed by external sources (e.g., advertising) and then take off through
    word-of-mouth or social contagion.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        args = [current_adopters, total_potential]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_orig'), object.__getattribute__(self, 'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_mutants'), args, kwargs, self)

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_orig(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_1(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = None
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_2(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get(None, 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_3(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", None)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_4(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get(0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_5(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", )
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_6(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("XXinnovation_coeffXX", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_7(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("INNOVATION_COEFF", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_8(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 1.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_9(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = None
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_10(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get(None, 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_11(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", None)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_12(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get(0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_13(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", )
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_14(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("XXimitation_coeffXX", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_15(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("IMITATION_COEFF", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_16(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 1.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_17(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = None
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_18(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = None
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_19(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) / (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_20(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p - q * (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_21(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q / (N / M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_22(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N * M)) * (M - N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_23(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M + N) if M > 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_24(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M >= 0 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_25(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 1 else 0

    def xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_26(self, current_adopters, total_potential, **params):
        """Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)

        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.

        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters

        Parameters
        ----------
                current_adopters (float): Current number of adopters.
                total_potential (float): Total potential number of adopters.

        Returns
        -------
                float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 1
    
    xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_1': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_1, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_2': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_2, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_3': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_3, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_4': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_4, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_5': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_5, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_6': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_6, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_7': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_7, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_8': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_8, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_9': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_9, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_10': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_10, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_11': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_11, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_12': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_12, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_13': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_13, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_14': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_14, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_15': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_15, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_16': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_16, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_17': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_17, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_18': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_18, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_19': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_19, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_20': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_20, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_21': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_21, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_22': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_22, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_23': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_23, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_24': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_24, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_25': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_25, 
        'xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_26': xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_26
    }
    xǁDualInfluenceGrowthǁcompute_growth_rate__mutmut_orig.__name__ = 'xǁDualInfluenceGrowthǁcompute_growth_rate'

    def predict_cumulative(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        args = [time_points, initial_adopters, total_potential]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_orig'), object.__getattribute__(self, 'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_mutants'), args, kwargs, self)

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_orig(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_1(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = None
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_2(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get(None, 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_3(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", None)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_4(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get(0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_5(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", )
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_6(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("XXinnovation_coeffXX", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_7(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("INNOVATION_COEFF", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_8(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 1.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_9(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = None
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_10(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get(None, 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_11(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", None)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_12(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get(0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_13(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", )
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_14(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("XXimitation_coeffXX", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_15(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("IMITATION_COEFF", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_16(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 1.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_17(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = None

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_18(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(None, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_19(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, None, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_20(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=None, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_21(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=None)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_22(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_23(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_24(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_25(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, )

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_26(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = None
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_27(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            None,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_28(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            None,
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_29(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            None,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_30(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=None,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_31(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method=None,
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_32(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_33(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_34(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_35(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_36(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_37(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[1], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_38(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[+1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_39(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-2]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_40(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="XXLSODAXX",
        )
        return sol.y.flatten()

    def xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_41(
        self,
        time_points,
        initial_adopters,
        total_potential,
        **params,
    ):
        """Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.

        Parameters
        ----------
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.

        Returns
        -------
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="lsoda",
        )
        return sol.y.flatten()
    
    xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_1': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_1, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_2': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_2, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_3': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_3, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_4': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_4, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_5': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_5, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_6': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_6, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_7': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_7, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_8': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_8, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_9': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_9, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_10': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_10, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_11': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_11, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_12': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_12, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_13': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_13, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_14': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_14, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_15': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_15, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_16': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_16, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_17': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_17, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_18': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_18, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_19': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_19, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_20': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_20, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_21': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_21, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_22': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_22, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_23': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_23, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_24': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_24, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_25': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_25, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_26': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_26, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_27': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_27, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_28': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_28, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_29': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_29, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_30': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_30, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_31': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_31, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_32': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_32, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_33': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_33, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_34': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_34, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_35': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_35, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_36': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_36, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_37': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_37, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_38': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_38, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_39': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_39, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_40': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_40, 
        'xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_41': xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_41
    }
    xǁDualInfluenceGrowthǁpredict_cumulative__mutmut_orig.__name__ = 'xǁDualInfluenceGrowthǁpredict_cumulative'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "XXinnovation_coeffXX": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "INNOVATION_COEFF": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "XXtypeXX": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "TYPE": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "XXfloatXX",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "FLOAT",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "XXdefaultXX": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "DEFAULT": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 1.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "XXdescriptionXX": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "DESCRIPTION": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "XXThe coefficient of innovation (external influence).XX",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "the coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_14(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "THE COEFFICIENT OF INNOVATION (EXTERNAL INFLUENCE).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_15(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "XXimitation_coeffXX": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_16(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "IMITATION_COEFF": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_17(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "XXtypeXX": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_18(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "TYPE": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_19(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "XXfloatXX",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_20(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "FLOAT",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_21(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "XXdefaultXX": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_22(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "DEFAULT": 0.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_23(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 1.1,
                "description": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_24(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "XXdescriptionXX": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_25(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "DESCRIPTION": "The coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_26(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "XXThe coefficient of imitation (internal influence).XX",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_27(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "the coefficient of imitation (internal influence).",
            },
        }

    def xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_28(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the model parameters for innovation and imitation coefficients.

        Returns
        -------
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence).",
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "THE COEFFICIENT OF IMITATION (INTERNAL INFLUENCE).",
            },
        }
    
    xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_1': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_1, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_2': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_2, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_3': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_3, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_4': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_4, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_5': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_5, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_6': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_6, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_7': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_7, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_8': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_8, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_9': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_9, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_10': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_10, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_11': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_11, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_12': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_12, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_13': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_13, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_14': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_14, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_15': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_15, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_16': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_16, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_17': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_17, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_18': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_18, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_19': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_19, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_20': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_20, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_21': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_21, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_22': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_22, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_23': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_23, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_24': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_24, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_25': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_25, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_26': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_26, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_27': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_27, 
        'xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_28': xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_28
    }
    xǁDualInfluenceGrowthǁget_parameters_schema__mutmut_orig.__name__ = 'xǁDualInfluenceGrowthǁget_parameters_schema'
