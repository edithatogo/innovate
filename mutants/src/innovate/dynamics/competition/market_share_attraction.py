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


class MarketShareAttraction(CompetitiveInteraction):
    """Determines market share based on relative attractiveness, which can be
    dynamically influenced by attributes (e.g., price, quality).
    """

    def compute_interaction_rates(self, **params):
        """Calculates the instantaneous interaction rates.

        This method is not implemented because the market share attraction model does not use instantaneous interaction rates.
        """
        # This model is not based on differential equations, so this method is not applicable.

    def predict_states(self, time_points, **params):
        args = [time_points]# type: ignore
        kwargs = {**params}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMarketShareAttractionǁpredict_states__mutmut_orig'), object.__getattribute__(self, 'xǁMarketShareAttractionǁpredict_states__mutmut_mutants'), args, kwargs, self)

    def xǁMarketShareAttractionǁpredict_states__mutmut_orig(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_1(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = None
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_2(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get(None, [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_3(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", None)
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_4(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get([])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_5(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", )
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_6(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("XXattractivenessXX", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_7(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("ATTRACTIVENESS", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_8(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_9(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError(None)

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_10(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("XXAttractiveness values must be provided.XX")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_11(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_12(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("ATTRACTIVENESS VALUES MUST BE PROVIDED.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_13(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = None

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_14(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(None)

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_15(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(None))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_16(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness != 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_17(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 1:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_18(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(None)

        return B.array(attractiveness) / total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_19(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) * total_attractiveness

    def xǁMarketShareAttractionǁpredict_states__mutmut_20(self, time_points, **params):
        """Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.

        Parameters
        ----------
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.

        Returns
        -------
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.

        Raises
        ------
            ValueError: If attractiveness values are not provided.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(None) / total_attractiveness
    
    xǁMarketShareAttractionǁpredict_states__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMarketShareAttractionǁpredict_states__mutmut_1': xǁMarketShareAttractionǁpredict_states__mutmut_1, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_2': xǁMarketShareAttractionǁpredict_states__mutmut_2, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_3': xǁMarketShareAttractionǁpredict_states__mutmut_3, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_4': xǁMarketShareAttractionǁpredict_states__mutmut_4, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_5': xǁMarketShareAttractionǁpredict_states__mutmut_5, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_6': xǁMarketShareAttractionǁpredict_states__mutmut_6, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_7': xǁMarketShareAttractionǁpredict_states__mutmut_7, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_8': xǁMarketShareAttractionǁpredict_states__mutmut_8, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_9': xǁMarketShareAttractionǁpredict_states__mutmut_9, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_10': xǁMarketShareAttractionǁpredict_states__mutmut_10, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_11': xǁMarketShareAttractionǁpredict_states__mutmut_11, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_12': xǁMarketShareAttractionǁpredict_states__mutmut_12, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_13': xǁMarketShareAttractionǁpredict_states__mutmut_13, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_14': xǁMarketShareAttractionǁpredict_states__mutmut_14, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_15': xǁMarketShareAttractionǁpredict_states__mutmut_15, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_16': xǁMarketShareAttractionǁpredict_states__mutmut_16, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_17': xǁMarketShareAttractionǁpredict_states__mutmut_17, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_18': xǁMarketShareAttractionǁpredict_states__mutmut_18, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_19': xǁMarketShareAttractionǁpredict_states__mutmut_19, 
        'xǁMarketShareAttractionǁpredict_states__mutmut_20': xǁMarketShareAttractionǁpredict_states__mutmut_20
    }
    xǁMarketShareAttractionǁpredict_states__mutmut_orig.__name__ = 'xǁMarketShareAttractionǁpredict_states'

    def get_parameters_schema(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMarketShareAttractionǁget_parameters_schema__mutmut_orig'), object.__getattribute__(self, 'xǁMarketShareAttractionǁget_parameters_schema__mutmut_mutants'), args, kwargs, self)

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_orig(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_1(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "XXattractivenessXX": {
                "type": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_2(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "ATTRACTIVENESS": {
                "type": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_3(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "XXtypeXX": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_4(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "TYPE": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_5(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "XXlistXX",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_6(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "LIST",
                "default": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_7(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "XXdefaultXX": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_8(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "DEFAULT": [],
                "description": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_9(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "XXdescriptionXX": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_10(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "DESCRIPTION": "A list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_11(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "description": "XXA list of attractiveness values for each competing entity.XX",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_12(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "description": "a list of attractiveness values for each competing entity.",
            },
        }

    def xǁMarketShareAttractionǁget_parameters_schema__mutmut_13(self):
        """Returns the schema for the model's parameters.

        Return a schema describing the expected parameters for the market share attraction model.

        Returns
        -------
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "description": "A LIST OF ATTRACTIVENESS VALUES FOR EACH COMPETING ENTITY.",
            },
        }
    
    xǁMarketShareAttractionǁget_parameters_schema__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMarketShareAttractionǁget_parameters_schema__mutmut_1': xǁMarketShareAttractionǁget_parameters_schema__mutmut_1, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_2': xǁMarketShareAttractionǁget_parameters_schema__mutmut_2, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_3': xǁMarketShareAttractionǁget_parameters_schema__mutmut_3, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_4': xǁMarketShareAttractionǁget_parameters_schema__mutmut_4, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_5': xǁMarketShareAttractionǁget_parameters_schema__mutmut_5, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_6': xǁMarketShareAttractionǁget_parameters_schema__mutmut_6, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_7': xǁMarketShareAttractionǁget_parameters_schema__mutmut_7, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_8': xǁMarketShareAttractionǁget_parameters_schema__mutmut_8, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_9': xǁMarketShareAttractionǁget_parameters_schema__mutmut_9, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_10': xǁMarketShareAttractionǁget_parameters_schema__mutmut_10, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_11': xǁMarketShareAttractionǁget_parameters_schema__mutmut_11, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_12': xǁMarketShareAttractionǁget_parameters_schema__mutmut_12, 
        'xǁMarketShareAttractionǁget_parameters_schema__mutmut_13': xǁMarketShareAttractionǁget_parameters_schema__mutmut_13
    }
    xǁMarketShareAttractionǁget_parameters_schema__mutmut_orig.__name__ = 'xǁMarketShareAttractionǁget_parameters_schema'
