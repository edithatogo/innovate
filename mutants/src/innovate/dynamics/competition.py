# src/innovate/dynamics/competition.py

from abc import ABC, abstractmethod
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


class CompetitiveInteraction(ABC):
    """Abstract base class for competitive interaction models."""

    @abstractmethod
    def compute_interaction_rate(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate."""


class LotkaVolterra(CompetitiveInteraction):
    """Implements the Lotka-Volterra competition model."""

    def compute_interaction_rate(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        args = [population1, population2]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_orig'), object.__getattribute__(self, 'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_mutants'), args, kwargs, self)

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_orig(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("alpha", 0.1)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_1(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = None
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_2(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get(None, 0.1)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_3(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("alpha", None)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_4(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get(0.1)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_5(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("alpha", )
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_6(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("XXalphaXX", 0.1)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_7(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("ALPHA", 0.1)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_8(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("alpha", 1.1)
        return alpha * population1 * population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_9(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("alpha", 0.1)
        return alpha * population1 / population2

    def xǁLotkaVolterraǁcompute_interaction_rate__mutmut_10(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the Lotka-Volterra model."""
        alpha = params.get("alpha", 0.1)
        return alpha / population1 * population2
    
    xǁLotkaVolterraǁcompute_interaction_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_1': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_1, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_2': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_2, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_3': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_3, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_4': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_4, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_5': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_5, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_6': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_6, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_7': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_7, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_8': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_8, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_9': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_9, 
        'xǁLotkaVolterraǁcompute_interaction_rate__mutmut_10': xǁLotkaVolterraǁcompute_interaction_rate__mutmut_10
    }
    xǁLotkaVolterraǁcompute_interaction_rate__mutmut_orig.__name__ = 'xǁLotkaVolterraǁcompute_interaction_rate'


class MarketShareAttraction(CompetitiveInteraction):
    """Implements the market share attraction model."""

    def compute_interaction_rate(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        args = [population1, population2]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_orig'), object.__getattribute__(self, 'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_mutants'), args, kwargs, self)

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_orig(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_1(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = None
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_2(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get(None, 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_3(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", None)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_4(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get(0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_5(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", )
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_6(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("XXattraction1XX", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_7(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("ATTRACTION1", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_8(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 1.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_9(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = None
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_10(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get(None, 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_11(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", None)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_12(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get(0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_13(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", )
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_14(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("XXattraction2XX", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_15(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("ATTRACTION2", 0.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_16(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", 1.1)
        return attraction1 * population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_17(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 + attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_18(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 / population1 - attraction2 * population2

    def xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_19(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the market share attraction model."""
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 / population2
    
    xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_1': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_1, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_2': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_2, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_3': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_3, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_4': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_4, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_5': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_5, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_6': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_6, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_7': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_7, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_8': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_8, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_9': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_9, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_10': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_10, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_11': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_11, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_12': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_12, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_13': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_13, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_14': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_14, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_15': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_15, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_16': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_16, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_17': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_17, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_18': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_18, 
        'xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_19': xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_19
    }
    xǁMarketShareAttractionǁcompute_interaction_rate__mutmut_orig.__name__ = 'xǁMarketShareAttractionǁcompute_interaction_rate'


class ReplicatorDynamics(CompetitiveInteraction):
    """Implements the replicator dynamics model."""

    def compute_interaction_rate(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        args = [population1, population2]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_orig'), object.__getattribute__(self, 'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_mutants'), args, kwargs, self)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_orig(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_1(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = None
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_2(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get(None, 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_3(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", None)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_4(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get(0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_5(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", )
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_6(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("XXfitness1XX", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_7(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("FITNESS1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_8(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 1.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_9(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = None
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_10(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get(None, 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_11(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", None)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_12(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get(0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_13(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", )
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_14(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("XXfitness2XX", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_15(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("FITNESS2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_16(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 1.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_17(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = None
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_18(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) * (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_19(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 - fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_20(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 / population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_21(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 / population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_22(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 - population2)
        return population1 * (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_23(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 / (fitness1 - average_fitness)

    def xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_24(
        self,
        population1: float,
        population2: float,
        **params,
    ):
        """Calculates the instantaneous interaction rate for the replicator dynamics model."""
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 + average_fitness)
    
    xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_1': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_1, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_2': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_2, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_3': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_3, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_4': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_4, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_5': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_5, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_6': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_6, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_7': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_7, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_8': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_8, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_9': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_9, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_10': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_10, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_11': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_11, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_12': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_12, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_13': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_13, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_14': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_14, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_15': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_15, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_16': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_16, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_17': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_17, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_18': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_18, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_19': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_19, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_20': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_20, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_21': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_21, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_22': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_22, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_23': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_23, 
        'xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_24': xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_24
    }
    xǁReplicatorDynamicsǁcompute_interaction_rate__mutmut_orig.__name__ = 'xǁReplicatorDynamicsǁcompute_interaction_rate'
