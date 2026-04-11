from collections.abc import Sequence

import numpy as np

from innovate import backend
from innovate.base.base import DiffusionModel
from innovate.dynamics.growth.dual_influence import DualInfluenceGrowth
from innovate.utils.validation import (
    validate_covariates,
    validate_covariates_dict,
    validate_float,
    validate_sequence_numeric,
    validate_time_series,
)
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


class BassModel(DiffusionModel):
    """Implementation of the Bass Diffusion Model.
    This is a wrapper around the DualInfluenceGrowth dynamics model.
    """

    def __init__(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        args = [covariates, t_event]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁ__init____mutmut_orig(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_1(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = None
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_2(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = None
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_3(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(None)
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_4(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_5(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = None
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_6(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(None, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_7(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, None)
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_8(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float("t_event")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_9(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, )
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_10(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "XXt_eventXX")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_11(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "T_EVENT")
        else:
            self.t_event = t_event
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_12(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = None
        self.growth_model = DualInfluenceGrowth()

    def xǁBassModelǁ__init____mutmut_13(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize the BassModel with optional covariates, a time event, and a DualInfluenceGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list if not provided.
            t_event (float, optional): The time of a structural break or event. If provided, the model will fit separate parameters for the periods before and after this time.
        """
        self._params: dict[str, float] = {}
        self.covariates = validate_covariates(covariates)
        if t_event is not None:
            self.t_event = validate_float(t_event, "t_event")
        else:
            self.t_event = t_event
        self.growth_model = None
    
    xǁBassModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁ__init____mutmut_1': xǁBassModelǁ__init____mutmut_1, 
        'xǁBassModelǁ__init____mutmut_2': xǁBassModelǁ__init____mutmut_2, 
        'xǁBassModelǁ__init____mutmut_3': xǁBassModelǁ__init____mutmut_3, 
        'xǁBassModelǁ__init____mutmut_4': xǁBassModelǁ__init____mutmut_4, 
        'xǁBassModelǁ__init____mutmut_5': xǁBassModelǁ__init____mutmut_5, 
        'xǁBassModelǁ__init____mutmut_6': xǁBassModelǁ__init____mutmut_6, 
        'xǁBassModelǁ__init____mutmut_7': xǁBassModelǁ__init____mutmut_7, 
        'xǁBassModelǁ__init____mutmut_8': xǁBassModelǁ__init____mutmut_8, 
        'xǁBassModelǁ__init____mutmut_9': xǁBassModelǁ__init____mutmut_9, 
        'xǁBassModelǁ__init____mutmut_10': xǁBassModelǁ__init____mutmut_10, 
        'xǁBassModelǁ__init____mutmut_11': xǁBassModelǁ__init____mutmut_11, 
        'xǁBassModelǁ__init____mutmut_12': xǁBassModelǁ__init____mutmut_12, 
        'xǁBassModelǁ__init____mutmut_13': xǁBassModelǁ__init____mutmut_13
    }
    xǁBassModelǁ__init____mutmut_orig.__name__ = 'xǁBassModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        """Return the list of parameter names for the Bass model, including base parameters and covariate-related coefficients.

        Returns
        -------
            names (Sequence[str]): List of parameter names, with covariate effects included if applicable.
        """
        names = ["p", "q", "m"]
        if self.t_event is not None:
            names.extend(["p_post", "q_post", "m_post"])
        for cov in self.covariates:
            names.extend([f"beta_p_{cov}", f"beta_q_{cov}", f"beta_m_{cov}"])
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = None

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(None, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, None, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, None, "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", None)

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", )

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "XXtXX", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "T", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "XXyXX")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "Y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = None
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "XXpXX": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "P": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 1.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "XXqXX": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "Q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 1.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "XXmXX": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "M": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) / 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(None) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 2.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                None,
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "XXp_postXX": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "P_POST": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 1.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "XXq_postXX": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_32(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "Q_POST": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_33(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 1.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_34(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "XXm_postXX": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_35(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "M_POST": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_36(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) / 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_37(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(None) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_38(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 2.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_39(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = None
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_40(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 1.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_41(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = None
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_42(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 1.0
            guesses[f"beta_m_{cov}"] = 0.0
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_43(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = None
        return guesses

    def xǁBassModelǁinitial_guesses__mutmut_44(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        guesses = {
            "p": 0.001,
            "q": 0.1,
            "m": np.max(y_arr) * 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "p_post": 0.001,
                    "q_post": 0.1,
                    "m_post": np.max(y_arr) * 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_p_{cov}"] = 0.0
            guesses[f"beta_q_{cov}"] = 0.0
            guesses[f"beta_m_{cov}"] = 1.0
        return guesses
    
    xǁBassModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁinitial_guesses__mutmut_1': xǁBassModelǁinitial_guesses__mutmut_1, 
        'xǁBassModelǁinitial_guesses__mutmut_2': xǁBassModelǁinitial_guesses__mutmut_2, 
        'xǁBassModelǁinitial_guesses__mutmut_3': xǁBassModelǁinitial_guesses__mutmut_3, 
        'xǁBassModelǁinitial_guesses__mutmut_4': xǁBassModelǁinitial_guesses__mutmut_4, 
        'xǁBassModelǁinitial_guesses__mutmut_5': xǁBassModelǁinitial_guesses__mutmut_5, 
        'xǁBassModelǁinitial_guesses__mutmut_6': xǁBassModelǁinitial_guesses__mutmut_6, 
        'xǁBassModelǁinitial_guesses__mutmut_7': xǁBassModelǁinitial_guesses__mutmut_7, 
        'xǁBassModelǁinitial_guesses__mutmut_8': xǁBassModelǁinitial_guesses__mutmut_8, 
        'xǁBassModelǁinitial_guesses__mutmut_9': xǁBassModelǁinitial_guesses__mutmut_9, 
        'xǁBassModelǁinitial_guesses__mutmut_10': xǁBassModelǁinitial_guesses__mutmut_10, 
        'xǁBassModelǁinitial_guesses__mutmut_11': xǁBassModelǁinitial_guesses__mutmut_11, 
        'xǁBassModelǁinitial_guesses__mutmut_12': xǁBassModelǁinitial_guesses__mutmut_12, 
        'xǁBassModelǁinitial_guesses__mutmut_13': xǁBassModelǁinitial_guesses__mutmut_13, 
        'xǁBassModelǁinitial_guesses__mutmut_14': xǁBassModelǁinitial_guesses__mutmut_14, 
        'xǁBassModelǁinitial_guesses__mutmut_15': xǁBassModelǁinitial_guesses__mutmut_15, 
        'xǁBassModelǁinitial_guesses__mutmut_16': xǁBassModelǁinitial_guesses__mutmut_16, 
        'xǁBassModelǁinitial_guesses__mutmut_17': xǁBassModelǁinitial_guesses__mutmut_17, 
        'xǁBassModelǁinitial_guesses__mutmut_18': xǁBassModelǁinitial_guesses__mutmut_18, 
        'xǁBassModelǁinitial_guesses__mutmut_19': xǁBassModelǁinitial_guesses__mutmut_19, 
        'xǁBassModelǁinitial_guesses__mutmut_20': xǁBassModelǁinitial_guesses__mutmut_20, 
        'xǁBassModelǁinitial_guesses__mutmut_21': xǁBassModelǁinitial_guesses__mutmut_21, 
        'xǁBassModelǁinitial_guesses__mutmut_22': xǁBassModelǁinitial_guesses__mutmut_22, 
        'xǁBassModelǁinitial_guesses__mutmut_23': xǁBassModelǁinitial_guesses__mutmut_23, 
        'xǁBassModelǁinitial_guesses__mutmut_24': xǁBassModelǁinitial_guesses__mutmut_24, 
        'xǁBassModelǁinitial_guesses__mutmut_25': xǁBassModelǁinitial_guesses__mutmut_25, 
        'xǁBassModelǁinitial_guesses__mutmut_26': xǁBassModelǁinitial_guesses__mutmut_26, 
        'xǁBassModelǁinitial_guesses__mutmut_27': xǁBassModelǁinitial_guesses__mutmut_27, 
        'xǁBassModelǁinitial_guesses__mutmut_28': xǁBassModelǁinitial_guesses__mutmut_28, 
        'xǁBassModelǁinitial_guesses__mutmut_29': xǁBassModelǁinitial_guesses__mutmut_29, 
        'xǁBassModelǁinitial_guesses__mutmut_30': xǁBassModelǁinitial_guesses__mutmut_30, 
        'xǁBassModelǁinitial_guesses__mutmut_31': xǁBassModelǁinitial_guesses__mutmut_31, 
        'xǁBassModelǁinitial_guesses__mutmut_32': xǁBassModelǁinitial_guesses__mutmut_32, 
        'xǁBassModelǁinitial_guesses__mutmut_33': xǁBassModelǁinitial_guesses__mutmut_33, 
        'xǁBassModelǁinitial_guesses__mutmut_34': xǁBassModelǁinitial_guesses__mutmut_34, 
        'xǁBassModelǁinitial_guesses__mutmut_35': xǁBassModelǁinitial_guesses__mutmut_35, 
        'xǁBassModelǁinitial_guesses__mutmut_36': xǁBassModelǁinitial_guesses__mutmut_36, 
        'xǁBassModelǁinitial_guesses__mutmut_37': xǁBassModelǁinitial_guesses__mutmut_37, 
        'xǁBassModelǁinitial_guesses__mutmut_38': xǁBassModelǁinitial_guesses__mutmut_38, 
        'xǁBassModelǁinitial_guesses__mutmut_39': xǁBassModelǁinitial_guesses__mutmut_39, 
        'xǁBassModelǁinitial_guesses__mutmut_40': xǁBassModelǁinitial_guesses__mutmut_40, 
        'xǁBassModelǁinitial_guesses__mutmut_41': xǁBassModelǁinitial_guesses__mutmut_41, 
        'xǁBassModelǁinitial_guesses__mutmut_42': xǁBassModelǁinitial_guesses__mutmut_42, 
        'xǁBassModelǁinitial_guesses__mutmut_43': xǁBassModelǁinitial_guesses__mutmut_43, 
        'xǁBassModelǁinitial_guesses__mutmut_44': xǁBassModelǁinitial_guesses__mutmut_44
    }
    xǁBassModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁBassModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = None

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(None, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, None, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, None, "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", None)

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", )

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "XXtXX", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "T", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "XXyXX")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "Y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = None
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "XXpXX": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "P": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1.000001, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 1.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "XXqXX": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "Q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1.000001, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 2.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "XXmXX": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "M": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(None), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                None,
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_28(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "XXp_postXX": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_29(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "P_POST": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_30(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1.000001, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_31(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 1.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_32(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "XXq_postXX": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_33(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "Q_POST": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_34(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1.000001, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_35(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 2.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_36(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "XXm_postXX": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_37(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "M_POST": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_38(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(None), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_39(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = None
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_40(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (+np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_41(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = None
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_42(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (+np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁBassModelǁbounds__mutmut_43(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = None
        return bounds

    def xǁBassModelǁbounds__mutmut_44(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Bass model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds. Base parameters "p", "q", and "m" have fixed bounds; covariate-related parameters are unbounded.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        bounds = {
            "p": (1e-6, 0.1),
            "q": (1e-6, 1.0),
            "m": (np.max(y_arr), np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "p_post": (1e-6, 0.1),
                    "q_post": (1e-6, 1.0),
                    "m_post": (np.max(y_arr), np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_p_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_q_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_m_{cov}"] = (+np.inf, np.inf)
        return bounds
    
    xǁBassModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁbounds__mutmut_1': xǁBassModelǁbounds__mutmut_1, 
        'xǁBassModelǁbounds__mutmut_2': xǁBassModelǁbounds__mutmut_2, 
        'xǁBassModelǁbounds__mutmut_3': xǁBassModelǁbounds__mutmut_3, 
        'xǁBassModelǁbounds__mutmut_4': xǁBassModelǁbounds__mutmut_4, 
        'xǁBassModelǁbounds__mutmut_5': xǁBassModelǁbounds__mutmut_5, 
        'xǁBassModelǁbounds__mutmut_6': xǁBassModelǁbounds__mutmut_6, 
        'xǁBassModelǁbounds__mutmut_7': xǁBassModelǁbounds__mutmut_7, 
        'xǁBassModelǁbounds__mutmut_8': xǁBassModelǁbounds__mutmut_8, 
        'xǁBassModelǁbounds__mutmut_9': xǁBassModelǁbounds__mutmut_9, 
        'xǁBassModelǁbounds__mutmut_10': xǁBassModelǁbounds__mutmut_10, 
        'xǁBassModelǁbounds__mutmut_11': xǁBassModelǁbounds__mutmut_11, 
        'xǁBassModelǁbounds__mutmut_12': xǁBassModelǁbounds__mutmut_12, 
        'xǁBassModelǁbounds__mutmut_13': xǁBassModelǁbounds__mutmut_13, 
        'xǁBassModelǁbounds__mutmut_14': xǁBassModelǁbounds__mutmut_14, 
        'xǁBassModelǁbounds__mutmut_15': xǁBassModelǁbounds__mutmut_15, 
        'xǁBassModelǁbounds__mutmut_16': xǁBassModelǁbounds__mutmut_16, 
        'xǁBassModelǁbounds__mutmut_17': xǁBassModelǁbounds__mutmut_17, 
        'xǁBassModelǁbounds__mutmut_18': xǁBassModelǁbounds__mutmut_18, 
        'xǁBassModelǁbounds__mutmut_19': xǁBassModelǁbounds__mutmut_19, 
        'xǁBassModelǁbounds__mutmut_20': xǁBassModelǁbounds__mutmut_20, 
        'xǁBassModelǁbounds__mutmut_21': xǁBassModelǁbounds__mutmut_21, 
        'xǁBassModelǁbounds__mutmut_22': xǁBassModelǁbounds__mutmut_22, 
        'xǁBassModelǁbounds__mutmut_23': xǁBassModelǁbounds__mutmut_23, 
        'xǁBassModelǁbounds__mutmut_24': xǁBassModelǁbounds__mutmut_24, 
        'xǁBassModelǁbounds__mutmut_25': xǁBassModelǁbounds__mutmut_25, 
        'xǁBassModelǁbounds__mutmut_26': xǁBassModelǁbounds__mutmut_26, 
        'xǁBassModelǁbounds__mutmut_27': xǁBassModelǁbounds__mutmut_27, 
        'xǁBassModelǁbounds__mutmut_28': xǁBassModelǁbounds__mutmut_28, 
        'xǁBassModelǁbounds__mutmut_29': xǁBassModelǁbounds__mutmut_29, 
        'xǁBassModelǁbounds__mutmut_30': xǁBassModelǁbounds__mutmut_30, 
        'xǁBassModelǁbounds__mutmut_31': xǁBassModelǁbounds__mutmut_31, 
        'xǁBassModelǁbounds__mutmut_32': xǁBassModelǁbounds__mutmut_32, 
        'xǁBassModelǁbounds__mutmut_33': xǁBassModelǁbounds__mutmut_33, 
        'xǁBassModelǁbounds__mutmut_34': xǁBassModelǁbounds__mutmut_34, 
        'xǁBassModelǁbounds__mutmut_35': xǁBassModelǁbounds__mutmut_35, 
        'xǁBassModelǁbounds__mutmut_36': xǁBassModelǁbounds__mutmut_36, 
        'xǁBassModelǁbounds__mutmut_37': xǁBassModelǁbounds__mutmut_37, 
        'xǁBassModelǁbounds__mutmut_38': xǁBassModelǁbounds__mutmut_38, 
        'xǁBassModelǁbounds__mutmut_39': xǁBassModelǁbounds__mutmut_39, 
        'xǁBassModelǁbounds__mutmut_40': xǁBassModelǁbounds__mutmut_40, 
        'xǁBassModelǁbounds__mutmut_41': xǁBassModelǁbounds__mutmut_41, 
        'xǁBassModelǁbounds__mutmut_42': xǁBassModelǁbounds__mutmut_42, 
        'xǁBassModelǁbounds__mutmut_43': xǁBassModelǁbounds__mutmut_43, 
        'xǁBassModelǁbounds__mutmut_44': xǁBassModelǁbounds__mutmut_44
    }
    xǁBassModelǁbounds__mutmut_orig.__name__ = 'xǁBassModelǁbounds'

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁpredict__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = None
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(None, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, None)
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric("t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, )
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "XXtXX")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "T")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) != 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 1:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError(None)

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("XXParameter 't' cannot be emptyXX")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("PARAMETER 'T' CANNOT BE EMPTY")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(None):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr <= 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 1):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError(None)

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("XXTime values (t) must be non-negativeXX")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("TIME VALUES (T) MUST BE NON-NEGATIVE")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError(None)

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = None

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(None, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, None, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, None) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_32(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, ) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_33(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_34(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = None

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_35(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1.000001

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_36(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = None
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_37(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = None
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_38(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) + set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_39(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(None) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_40(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(None)
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_41(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(None)

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_42(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = None

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_43(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(None):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_44(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(None, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_45(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, None)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_46(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_47(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, )):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_48(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_49(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(None):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_50(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(None)

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_51(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(None, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_52(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, None, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_53(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, None, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_54(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, None, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_55(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, None)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_56(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_57(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_58(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_59(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_60(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, )

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_61(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = None
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_62(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(None, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_63(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, None, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_64(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, None, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_65(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, None)
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_66(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_67(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_68(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_69(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, )
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_70(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(None))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_71(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(None, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_72(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, None, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_73(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, None, validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_74(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), None, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_75(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, None)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_76(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_77(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_78(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_79(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_80(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, )

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_81(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(None), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_82(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = None
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_83(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(None, y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_84(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, None, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_85(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, None)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_86(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(y0, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_87(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, t_arr)
        return sol.flatten()

    def xǁBassModelǁpredict__mutmut_88(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        """Predicts cumulative adoption over time using the Bass diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption at each time point in `t`.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative (reasonable for diffusion models)
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y0 = 1e-6

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        # Note: Using the currently configured backend instead of forcing JAX

        # Validate that required parameters are present
        required_params = self.param_names
        missing_params = set(required_params) - set(self._params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in model: {missing_params}")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for i, (param_name, param_val) in enumerate(zip(required_params, params)):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        def ode_func(t, y, args):
            return self.differential_equation(t, y, args, validated_covariates, t)

        # Handle different backend method signatures
        from innovate.backends.jax_backend import JaxBackend

        if isinstance(backend.current_backend, JaxBackend):
            # JAX backend expects 4 parameters: func, y0, t, args
            sol = backend.current_backend.solve_ode(ode_func, y0, t_arr, tuple(params))
        else:
            # NumPy backend expects 3 parameters: func, y0, t (parameters must be in closure)
            # Modify the function to not require additional args
            def ode_func_numpy(y_val, t_val):
                return self.differential_equation(t_val, y_val, tuple(params), validated_covariates, t_arr)

            sol = backend.current_backend.solve_ode(ode_func_numpy, y0, )
        return sol.flatten()
    
    xǁBassModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁpredict__mutmut_1': xǁBassModelǁpredict__mutmut_1, 
        'xǁBassModelǁpredict__mutmut_2': xǁBassModelǁpredict__mutmut_2, 
        'xǁBassModelǁpredict__mutmut_3': xǁBassModelǁpredict__mutmut_3, 
        'xǁBassModelǁpredict__mutmut_4': xǁBassModelǁpredict__mutmut_4, 
        'xǁBassModelǁpredict__mutmut_5': xǁBassModelǁpredict__mutmut_5, 
        'xǁBassModelǁpredict__mutmut_6': xǁBassModelǁpredict__mutmut_6, 
        'xǁBassModelǁpredict__mutmut_7': xǁBassModelǁpredict__mutmut_7, 
        'xǁBassModelǁpredict__mutmut_8': xǁBassModelǁpredict__mutmut_8, 
        'xǁBassModelǁpredict__mutmut_9': xǁBassModelǁpredict__mutmut_9, 
        'xǁBassModelǁpredict__mutmut_10': xǁBassModelǁpredict__mutmut_10, 
        'xǁBassModelǁpredict__mutmut_11': xǁBassModelǁpredict__mutmut_11, 
        'xǁBassModelǁpredict__mutmut_12': xǁBassModelǁpredict__mutmut_12, 
        'xǁBassModelǁpredict__mutmut_13': xǁBassModelǁpredict__mutmut_13, 
        'xǁBassModelǁpredict__mutmut_14': xǁBassModelǁpredict__mutmut_14, 
        'xǁBassModelǁpredict__mutmut_15': xǁBassModelǁpredict__mutmut_15, 
        'xǁBassModelǁpredict__mutmut_16': xǁBassModelǁpredict__mutmut_16, 
        'xǁBassModelǁpredict__mutmut_17': xǁBassModelǁpredict__mutmut_17, 
        'xǁBassModelǁpredict__mutmut_18': xǁBassModelǁpredict__mutmut_18, 
        'xǁBassModelǁpredict__mutmut_19': xǁBassModelǁpredict__mutmut_19, 
        'xǁBassModelǁpredict__mutmut_20': xǁBassModelǁpredict__mutmut_20, 
        'xǁBassModelǁpredict__mutmut_21': xǁBassModelǁpredict__mutmut_21, 
        'xǁBassModelǁpredict__mutmut_22': xǁBassModelǁpredict__mutmut_22, 
        'xǁBassModelǁpredict__mutmut_23': xǁBassModelǁpredict__mutmut_23, 
        'xǁBassModelǁpredict__mutmut_24': xǁBassModelǁpredict__mutmut_24, 
        'xǁBassModelǁpredict__mutmut_25': xǁBassModelǁpredict__mutmut_25, 
        'xǁBassModelǁpredict__mutmut_26': xǁBassModelǁpredict__mutmut_26, 
        'xǁBassModelǁpredict__mutmut_27': xǁBassModelǁpredict__mutmut_27, 
        'xǁBassModelǁpredict__mutmut_28': xǁBassModelǁpredict__mutmut_28, 
        'xǁBassModelǁpredict__mutmut_29': xǁBassModelǁpredict__mutmut_29, 
        'xǁBassModelǁpredict__mutmut_30': xǁBassModelǁpredict__mutmut_30, 
        'xǁBassModelǁpredict__mutmut_31': xǁBassModelǁpredict__mutmut_31, 
        'xǁBassModelǁpredict__mutmut_32': xǁBassModelǁpredict__mutmut_32, 
        'xǁBassModelǁpredict__mutmut_33': xǁBassModelǁpredict__mutmut_33, 
        'xǁBassModelǁpredict__mutmut_34': xǁBassModelǁpredict__mutmut_34, 
        'xǁBassModelǁpredict__mutmut_35': xǁBassModelǁpredict__mutmut_35, 
        'xǁBassModelǁpredict__mutmut_36': xǁBassModelǁpredict__mutmut_36, 
        'xǁBassModelǁpredict__mutmut_37': xǁBassModelǁpredict__mutmut_37, 
        'xǁBassModelǁpredict__mutmut_38': xǁBassModelǁpredict__mutmut_38, 
        'xǁBassModelǁpredict__mutmut_39': xǁBassModelǁpredict__mutmut_39, 
        'xǁBassModelǁpredict__mutmut_40': xǁBassModelǁpredict__mutmut_40, 
        'xǁBassModelǁpredict__mutmut_41': xǁBassModelǁpredict__mutmut_41, 
        'xǁBassModelǁpredict__mutmut_42': xǁBassModelǁpredict__mutmut_42, 
        'xǁBassModelǁpredict__mutmut_43': xǁBassModelǁpredict__mutmut_43, 
        'xǁBassModelǁpredict__mutmut_44': xǁBassModelǁpredict__mutmut_44, 
        'xǁBassModelǁpredict__mutmut_45': xǁBassModelǁpredict__mutmut_45, 
        'xǁBassModelǁpredict__mutmut_46': xǁBassModelǁpredict__mutmut_46, 
        'xǁBassModelǁpredict__mutmut_47': xǁBassModelǁpredict__mutmut_47, 
        'xǁBassModelǁpredict__mutmut_48': xǁBassModelǁpredict__mutmut_48, 
        'xǁBassModelǁpredict__mutmut_49': xǁBassModelǁpredict__mutmut_49, 
        'xǁBassModelǁpredict__mutmut_50': xǁBassModelǁpredict__mutmut_50, 
        'xǁBassModelǁpredict__mutmut_51': xǁBassModelǁpredict__mutmut_51, 
        'xǁBassModelǁpredict__mutmut_52': xǁBassModelǁpredict__mutmut_52, 
        'xǁBassModelǁpredict__mutmut_53': xǁBassModelǁpredict__mutmut_53, 
        'xǁBassModelǁpredict__mutmut_54': xǁBassModelǁpredict__mutmut_54, 
        'xǁBassModelǁpredict__mutmut_55': xǁBassModelǁpredict__mutmut_55, 
        'xǁBassModelǁpredict__mutmut_56': xǁBassModelǁpredict__mutmut_56, 
        'xǁBassModelǁpredict__mutmut_57': xǁBassModelǁpredict__mutmut_57, 
        'xǁBassModelǁpredict__mutmut_58': xǁBassModelǁpredict__mutmut_58, 
        'xǁBassModelǁpredict__mutmut_59': xǁBassModelǁpredict__mutmut_59, 
        'xǁBassModelǁpredict__mutmut_60': xǁBassModelǁpredict__mutmut_60, 
        'xǁBassModelǁpredict__mutmut_61': xǁBassModelǁpredict__mutmut_61, 
        'xǁBassModelǁpredict__mutmut_62': xǁBassModelǁpredict__mutmut_62, 
        'xǁBassModelǁpredict__mutmut_63': xǁBassModelǁpredict__mutmut_63, 
        'xǁBassModelǁpredict__mutmut_64': xǁBassModelǁpredict__mutmut_64, 
        'xǁBassModelǁpredict__mutmut_65': xǁBassModelǁpredict__mutmut_65, 
        'xǁBassModelǁpredict__mutmut_66': xǁBassModelǁpredict__mutmut_66, 
        'xǁBassModelǁpredict__mutmut_67': xǁBassModelǁpredict__mutmut_67, 
        'xǁBassModelǁpredict__mutmut_68': xǁBassModelǁpredict__mutmut_68, 
        'xǁBassModelǁpredict__mutmut_69': xǁBassModelǁpredict__mutmut_69, 
        'xǁBassModelǁpredict__mutmut_70': xǁBassModelǁpredict__mutmut_70, 
        'xǁBassModelǁpredict__mutmut_71': xǁBassModelǁpredict__mutmut_71, 
        'xǁBassModelǁpredict__mutmut_72': xǁBassModelǁpredict__mutmut_72, 
        'xǁBassModelǁpredict__mutmut_73': xǁBassModelǁpredict__mutmut_73, 
        'xǁBassModelǁpredict__mutmut_74': xǁBassModelǁpredict__mutmut_74, 
        'xǁBassModelǁpredict__mutmut_75': xǁBassModelǁpredict__mutmut_75, 
        'xǁBassModelǁpredict__mutmut_76': xǁBassModelǁpredict__mutmut_76, 
        'xǁBassModelǁpredict__mutmut_77': xǁBassModelǁpredict__mutmut_77, 
        'xǁBassModelǁpredict__mutmut_78': xǁBassModelǁpredict__mutmut_78, 
        'xǁBassModelǁpredict__mutmut_79': xǁBassModelǁpredict__mutmut_79, 
        'xǁBassModelǁpredict__mutmut_80': xǁBassModelǁpredict__mutmut_80, 
        'xǁBassModelǁpredict__mutmut_81': xǁBassModelǁpredict__mutmut_81, 
        'xǁBassModelǁpredict__mutmut_82': xǁBassModelǁpredict__mutmut_82, 
        'xǁBassModelǁpredict__mutmut_83': xǁBassModelǁpredict__mutmut_83, 
        'xǁBassModelǁpredict__mutmut_84': xǁBassModelǁpredict__mutmut_84, 
        'xǁBassModelǁpredict__mutmut_85': xǁBassModelǁpredict__mutmut_85, 
        'xǁBassModelǁpredict__mutmut_86': xǁBassModelǁpredict__mutmut_86, 
        'xǁBassModelǁpredict__mutmut_87': xǁBassModelǁpredict__mutmut_87, 
        'xǁBassModelǁpredict__mutmut_88': xǁBassModelǁpredict__mutmut_88
    }
    xǁBassModelǁpredict__mutmut_orig.__name__ = 'xǁBassModelǁpredict'

    def differential_equation(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        args = [t, y, params, covariates, t_eval]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁdifferential_equation__mutmut_orig(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_1(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None or t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_2(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_3(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t > self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_4(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = None
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_5(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[4]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_6(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = None
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_7(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[5]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_8(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = None
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_9(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[6]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_10(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = None
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_11(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 4
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_12(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = None
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_13(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[1]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_14(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = None
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_15(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[2]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_16(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = None
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_17(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[3]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_18(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = None

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_19(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 1

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_20(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = None
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_21(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = None
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_22(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = None

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_23(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = None
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_24(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 - param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_25(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 4 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_26(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = None

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_27(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(None, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_28(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, None, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_29(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, None)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_30(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_31(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_32(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, )

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_33(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t = params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_34(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t -= params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_35(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] / cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_36(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t = params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_37(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t -= params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_38(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] / cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_39(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx - 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_40(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 2] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_41(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t = params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_42(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t -= params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_43(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] / cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_44(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx - 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_45(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 3] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_46(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx = 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_47(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx -= 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_48(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 4

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_49(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = None
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_50(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) / (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_51(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t - q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_52(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t / (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_53(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y * m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_54(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t + y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_55(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(None, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_56(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, None, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_57(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, None)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_58(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_59(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_60(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, )
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_61(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t >= 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_62(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 1, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_63(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 1.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_64(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(None, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_65(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, None, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_66(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, None)

    def xǁBassModelǁdifferential_equation__mutmut_67(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_68(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_69(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, )

    def xǁBassModelǁdifferential_equation__mutmut_70(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t >= 0, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_71(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 1, rate, 0.0)

    def xǁBassModelǁdifferential_equation__mutmut_72(
        self,
        t: float,
        y: float,
        params: Sequence[float],
        covariates: dict[str, Sequence[float]] | None,
        t_eval: Sequence[float],
    ) -> float:
        """Defines the Bass model's differential equation, incorporating covariate effects if provided.

        At each time point, adjusts the innovation, imitation, and market size parameters by linearly combining base values with covariate contributions, then computes the instantaneous growth rate using the underlying DualInfluenceGrowth model.

        Parameters
        ----------
            t: Current time point.
            y: Current cumulative adoption value.
            params: Sequence of model parameters, including base and covariate coefficients.
            covariates: Optional dictionary mapping covariate names to their time series values.
            t_eval: Sequence of time points for covariate interpolation.

        Returns
        -------
            The instantaneous adoption rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            p_base = params[3]
            q_base = params[4]
            m_base = params[5]
            param_idx_offset = 3
        else:
            p_base = params[0]
            q_base = params[1]
            m_base = params[2]
            param_idx_offset = 0

        p_t = p_base
        q_t = q_base
        m_t = m_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)

                p_t += params[param_idx] * cov_val_t
                q_t += params[param_idx + 1] * cov_val_t
                m_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        rate = (p_t + q_t * (y / m_t)) * (m_t - y)
        try:
            import pytensor.tensor as pt  # type: ignore

            if isinstance(
                m_t,
                pt.TensorVariable,
            ):  # pragma: no cover - depends on pytensor
                return pt.switch(m_t > 0, rate, 0.0)
        except Exception:
            pass
        return backend.current_backend.where(m_t > 0, rate, 1.0)
    
    xǁBassModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁdifferential_equation__mutmut_1': xǁBassModelǁdifferential_equation__mutmut_1, 
        'xǁBassModelǁdifferential_equation__mutmut_2': xǁBassModelǁdifferential_equation__mutmut_2, 
        'xǁBassModelǁdifferential_equation__mutmut_3': xǁBassModelǁdifferential_equation__mutmut_3, 
        'xǁBassModelǁdifferential_equation__mutmut_4': xǁBassModelǁdifferential_equation__mutmut_4, 
        'xǁBassModelǁdifferential_equation__mutmut_5': xǁBassModelǁdifferential_equation__mutmut_5, 
        'xǁBassModelǁdifferential_equation__mutmut_6': xǁBassModelǁdifferential_equation__mutmut_6, 
        'xǁBassModelǁdifferential_equation__mutmut_7': xǁBassModelǁdifferential_equation__mutmut_7, 
        'xǁBassModelǁdifferential_equation__mutmut_8': xǁBassModelǁdifferential_equation__mutmut_8, 
        'xǁBassModelǁdifferential_equation__mutmut_9': xǁBassModelǁdifferential_equation__mutmut_9, 
        'xǁBassModelǁdifferential_equation__mutmut_10': xǁBassModelǁdifferential_equation__mutmut_10, 
        'xǁBassModelǁdifferential_equation__mutmut_11': xǁBassModelǁdifferential_equation__mutmut_11, 
        'xǁBassModelǁdifferential_equation__mutmut_12': xǁBassModelǁdifferential_equation__mutmut_12, 
        'xǁBassModelǁdifferential_equation__mutmut_13': xǁBassModelǁdifferential_equation__mutmut_13, 
        'xǁBassModelǁdifferential_equation__mutmut_14': xǁBassModelǁdifferential_equation__mutmut_14, 
        'xǁBassModelǁdifferential_equation__mutmut_15': xǁBassModelǁdifferential_equation__mutmut_15, 
        'xǁBassModelǁdifferential_equation__mutmut_16': xǁBassModelǁdifferential_equation__mutmut_16, 
        'xǁBassModelǁdifferential_equation__mutmut_17': xǁBassModelǁdifferential_equation__mutmut_17, 
        'xǁBassModelǁdifferential_equation__mutmut_18': xǁBassModelǁdifferential_equation__mutmut_18, 
        'xǁBassModelǁdifferential_equation__mutmut_19': xǁBassModelǁdifferential_equation__mutmut_19, 
        'xǁBassModelǁdifferential_equation__mutmut_20': xǁBassModelǁdifferential_equation__mutmut_20, 
        'xǁBassModelǁdifferential_equation__mutmut_21': xǁBassModelǁdifferential_equation__mutmut_21, 
        'xǁBassModelǁdifferential_equation__mutmut_22': xǁBassModelǁdifferential_equation__mutmut_22, 
        'xǁBassModelǁdifferential_equation__mutmut_23': xǁBassModelǁdifferential_equation__mutmut_23, 
        'xǁBassModelǁdifferential_equation__mutmut_24': xǁBassModelǁdifferential_equation__mutmut_24, 
        'xǁBassModelǁdifferential_equation__mutmut_25': xǁBassModelǁdifferential_equation__mutmut_25, 
        'xǁBassModelǁdifferential_equation__mutmut_26': xǁBassModelǁdifferential_equation__mutmut_26, 
        'xǁBassModelǁdifferential_equation__mutmut_27': xǁBassModelǁdifferential_equation__mutmut_27, 
        'xǁBassModelǁdifferential_equation__mutmut_28': xǁBassModelǁdifferential_equation__mutmut_28, 
        'xǁBassModelǁdifferential_equation__mutmut_29': xǁBassModelǁdifferential_equation__mutmut_29, 
        'xǁBassModelǁdifferential_equation__mutmut_30': xǁBassModelǁdifferential_equation__mutmut_30, 
        'xǁBassModelǁdifferential_equation__mutmut_31': xǁBassModelǁdifferential_equation__mutmut_31, 
        'xǁBassModelǁdifferential_equation__mutmut_32': xǁBassModelǁdifferential_equation__mutmut_32, 
        'xǁBassModelǁdifferential_equation__mutmut_33': xǁBassModelǁdifferential_equation__mutmut_33, 
        'xǁBassModelǁdifferential_equation__mutmut_34': xǁBassModelǁdifferential_equation__mutmut_34, 
        'xǁBassModelǁdifferential_equation__mutmut_35': xǁBassModelǁdifferential_equation__mutmut_35, 
        'xǁBassModelǁdifferential_equation__mutmut_36': xǁBassModelǁdifferential_equation__mutmut_36, 
        'xǁBassModelǁdifferential_equation__mutmut_37': xǁBassModelǁdifferential_equation__mutmut_37, 
        'xǁBassModelǁdifferential_equation__mutmut_38': xǁBassModelǁdifferential_equation__mutmut_38, 
        'xǁBassModelǁdifferential_equation__mutmut_39': xǁBassModelǁdifferential_equation__mutmut_39, 
        'xǁBassModelǁdifferential_equation__mutmut_40': xǁBassModelǁdifferential_equation__mutmut_40, 
        'xǁBassModelǁdifferential_equation__mutmut_41': xǁBassModelǁdifferential_equation__mutmut_41, 
        'xǁBassModelǁdifferential_equation__mutmut_42': xǁBassModelǁdifferential_equation__mutmut_42, 
        'xǁBassModelǁdifferential_equation__mutmut_43': xǁBassModelǁdifferential_equation__mutmut_43, 
        'xǁBassModelǁdifferential_equation__mutmut_44': xǁBassModelǁdifferential_equation__mutmut_44, 
        'xǁBassModelǁdifferential_equation__mutmut_45': xǁBassModelǁdifferential_equation__mutmut_45, 
        'xǁBassModelǁdifferential_equation__mutmut_46': xǁBassModelǁdifferential_equation__mutmut_46, 
        'xǁBassModelǁdifferential_equation__mutmut_47': xǁBassModelǁdifferential_equation__mutmut_47, 
        'xǁBassModelǁdifferential_equation__mutmut_48': xǁBassModelǁdifferential_equation__mutmut_48, 
        'xǁBassModelǁdifferential_equation__mutmut_49': xǁBassModelǁdifferential_equation__mutmut_49, 
        'xǁBassModelǁdifferential_equation__mutmut_50': xǁBassModelǁdifferential_equation__mutmut_50, 
        'xǁBassModelǁdifferential_equation__mutmut_51': xǁBassModelǁdifferential_equation__mutmut_51, 
        'xǁBassModelǁdifferential_equation__mutmut_52': xǁBassModelǁdifferential_equation__mutmut_52, 
        'xǁBassModelǁdifferential_equation__mutmut_53': xǁBassModelǁdifferential_equation__mutmut_53, 
        'xǁBassModelǁdifferential_equation__mutmut_54': xǁBassModelǁdifferential_equation__mutmut_54, 
        'xǁBassModelǁdifferential_equation__mutmut_55': xǁBassModelǁdifferential_equation__mutmut_55, 
        'xǁBassModelǁdifferential_equation__mutmut_56': xǁBassModelǁdifferential_equation__mutmut_56, 
        'xǁBassModelǁdifferential_equation__mutmut_57': xǁBassModelǁdifferential_equation__mutmut_57, 
        'xǁBassModelǁdifferential_equation__mutmut_58': xǁBassModelǁdifferential_equation__mutmut_58, 
        'xǁBassModelǁdifferential_equation__mutmut_59': xǁBassModelǁdifferential_equation__mutmut_59, 
        'xǁBassModelǁdifferential_equation__mutmut_60': xǁBassModelǁdifferential_equation__mutmut_60, 
        'xǁBassModelǁdifferential_equation__mutmut_61': xǁBassModelǁdifferential_equation__mutmut_61, 
        'xǁBassModelǁdifferential_equation__mutmut_62': xǁBassModelǁdifferential_equation__mutmut_62, 
        'xǁBassModelǁdifferential_equation__mutmut_63': xǁBassModelǁdifferential_equation__mutmut_63, 
        'xǁBassModelǁdifferential_equation__mutmut_64': xǁBassModelǁdifferential_equation__mutmut_64, 
        'xǁBassModelǁdifferential_equation__mutmut_65': xǁBassModelǁdifferential_equation__mutmut_65, 
        'xǁBassModelǁdifferential_equation__mutmut_66': xǁBassModelǁdifferential_equation__mutmut_66, 
        'xǁBassModelǁdifferential_equation__mutmut_67': xǁBassModelǁdifferential_equation__mutmut_67, 
        'xǁBassModelǁdifferential_equation__mutmut_68': xǁBassModelǁdifferential_equation__mutmut_68, 
        'xǁBassModelǁdifferential_equation__mutmut_69': xǁBassModelǁdifferential_equation__mutmut_69, 
        'xǁBassModelǁdifferential_equation__mutmut_70': xǁBassModelǁdifferential_equation__mutmut_70, 
        'xǁBassModelǁdifferential_equation__mutmut_71': xǁBassModelǁdifferential_equation__mutmut_71, 
        'xǁBassModelǁdifferential_equation__mutmut_72': xǁBassModelǁdifferential_equation__mutmut_72
    }
    xǁBassModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁBassModelǁdifferential_equation'

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        args = [t, y, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁscore__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = None

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(None, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, None, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, None, "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", None)

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", )

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "XXtXX", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "T", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "XXyXX")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "Y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError(None)

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = None

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(None, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, None, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, None) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, ) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = None

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(None, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, None)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, )

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_32(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = None
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_33(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(None)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_34(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_35(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(None):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_36(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(None)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_37(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError(None)

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_38(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("XXPrediction resulted in non-finite valuesXX")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_39(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_40(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("PREDICTION RESULTED IN NON-FINITE VALUES")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_41(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = None
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_42(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            None,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_43(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) * 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_44(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) + y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_45(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(None) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_46(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 3,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_47(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_48(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            None,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_49(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) * 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_50(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) + backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_51(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(None) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_52(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(None)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_53(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 3,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_54(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_55(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_56(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁBassModelǁscore__mutmut_57(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁBassModelǁscore__mutmut_58(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁBassModelǁscore__mutmut_59(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed and predicted values.

        Parameters
        ----------
            t (Sequence[float]): Sequence of time points.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Optional time series of covariate values affecting model parameters.

        Returns
        -------
                float: R² score indicating the proportion of variance explained by the model predictions.
        """
        # Validate inputs
        t_arr, y_arr = validate_time_series(t, y, "t", "y")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y_arr) - backend.current_backend.mean(y_arr)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁBassModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁscore__mutmut_1': xǁBassModelǁscore__mutmut_1, 
        'xǁBassModelǁscore__mutmut_2': xǁBassModelǁscore__mutmut_2, 
        'xǁBassModelǁscore__mutmut_3': xǁBassModelǁscore__mutmut_3, 
        'xǁBassModelǁscore__mutmut_4': xǁBassModelǁscore__mutmut_4, 
        'xǁBassModelǁscore__mutmut_5': xǁBassModelǁscore__mutmut_5, 
        'xǁBassModelǁscore__mutmut_6': xǁBassModelǁscore__mutmut_6, 
        'xǁBassModelǁscore__mutmut_7': xǁBassModelǁscore__mutmut_7, 
        'xǁBassModelǁscore__mutmut_8': xǁBassModelǁscore__mutmut_8, 
        'xǁBassModelǁscore__mutmut_9': xǁBassModelǁscore__mutmut_9, 
        'xǁBassModelǁscore__mutmut_10': xǁBassModelǁscore__mutmut_10, 
        'xǁBassModelǁscore__mutmut_11': xǁBassModelǁscore__mutmut_11, 
        'xǁBassModelǁscore__mutmut_12': xǁBassModelǁscore__mutmut_12, 
        'xǁBassModelǁscore__mutmut_13': xǁBassModelǁscore__mutmut_13, 
        'xǁBassModelǁscore__mutmut_14': xǁBassModelǁscore__mutmut_14, 
        'xǁBassModelǁscore__mutmut_15': xǁBassModelǁscore__mutmut_15, 
        'xǁBassModelǁscore__mutmut_16': xǁBassModelǁscore__mutmut_16, 
        'xǁBassModelǁscore__mutmut_17': xǁBassModelǁscore__mutmut_17, 
        'xǁBassModelǁscore__mutmut_18': xǁBassModelǁscore__mutmut_18, 
        'xǁBassModelǁscore__mutmut_19': xǁBassModelǁscore__mutmut_19, 
        'xǁBassModelǁscore__mutmut_20': xǁBassModelǁscore__mutmut_20, 
        'xǁBassModelǁscore__mutmut_21': xǁBassModelǁscore__mutmut_21, 
        'xǁBassModelǁscore__mutmut_22': xǁBassModelǁscore__mutmut_22, 
        'xǁBassModelǁscore__mutmut_23': xǁBassModelǁscore__mutmut_23, 
        'xǁBassModelǁscore__mutmut_24': xǁBassModelǁscore__mutmut_24, 
        'xǁBassModelǁscore__mutmut_25': xǁBassModelǁscore__mutmut_25, 
        'xǁBassModelǁscore__mutmut_26': xǁBassModelǁscore__mutmut_26, 
        'xǁBassModelǁscore__mutmut_27': xǁBassModelǁscore__mutmut_27, 
        'xǁBassModelǁscore__mutmut_28': xǁBassModelǁscore__mutmut_28, 
        'xǁBassModelǁscore__mutmut_29': xǁBassModelǁscore__mutmut_29, 
        'xǁBassModelǁscore__mutmut_30': xǁBassModelǁscore__mutmut_30, 
        'xǁBassModelǁscore__mutmut_31': xǁBassModelǁscore__mutmut_31, 
        'xǁBassModelǁscore__mutmut_32': xǁBassModelǁscore__mutmut_32, 
        'xǁBassModelǁscore__mutmut_33': xǁBassModelǁscore__mutmut_33, 
        'xǁBassModelǁscore__mutmut_34': xǁBassModelǁscore__mutmut_34, 
        'xǁBassModelǁscore__mutmut_35': xǁBassModelǁscore__mutmut_35, 
        'xǁBassModelǁscore__mutmut_36': xǁBassModelǁscore__mutmut_36, 
        'xǁBassModelǁscore__mutmut_37': xǁBassModelǁscore__mutmut_37, 
        'xǁBassModelǁscore__mutmut_38': xǁBassModelǁscore__mutmut_38, 
        'xǁBassModelǁscore__mutmut_39': xǁBassModelǁscore__mutmut_39, 
        'xǁBassModelǁscore__mutmut_40': xǁBassModelǁscore__mutmut_40, 
        'xǁBassModelǁscore__mutmut_41': xǁBassModelǁscore__mutmut_41, 
        'xǁBassModelǁscore__mutmut_42': xǁBassModelǁscore__mutmut_42, 
        'xǁBassModelǁscore__mutmut_43': xǁBassModelǁscore__mutmut_43, 
        'xǁBassModelǁscore__mutmut_44': xǁBassModelǁscore__mutmut_44, 
        'xǁBassModelǁscore__mutmut_45': xǁBassModelǁscore__mutmut_45, 
        'xǁBassModelǁscore__mutmut_46': xǁBassModelǁscore__mutmut_46, 
        'xǁBassModelǁscore__mutmut_47': xǁBassModelǁscore__mutmut_47, 
        'xǁBassModelǁscore__mutmut_48': xǁBassModelǁscore__mutmut_48, 
        'xǁBassModelǁscore__mutmut_49': xǁBassModelǁscore__mutmut_49, 
        'xǁBassModelǁscore__mutmut_50': xǁBassModelǁscore__mutmut_50, 
        'xǁBassModelǁscore__mutmut_51': xǁBassModelǁscore__mutmut_51, 
        'xǁBassModelǁscore__mutmut_52': xǁBassModelǁscore__mutmut_52, 
        'xǁBassModelǁscore__mutmut_53': xǁBassModelǁscore__mutmut_53, 
        'xǁBassModelǁscore__mutmut_54': xǁBassModelǁscore__mutmut_54, 
        'xǁBassModelǁscore__mutmut_55': xǁBassModelǁscore__mutmut_55, 
        'xǁBassModelǁscore__mutmut_56': xǁBassModelǁscore__mutmut_56, 
        'xǁBassModelǁscore__mutmut_57': xǁBassModelǁscore__mutmut_57, 
        'xǁBassModelǁscore__mutmut_58': xǁBassModelǁscore__mutmut_58, 
        'xǁBassModelǁscore__mutmut_59': xǁBassModelǁscore__mutmut_59
    }
    xǁBassModelǁscore__mutmut_orig.__name__ = 'xǁBassModelǁscore'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = None
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(None, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, None)
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric("t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, )
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "XXtXX")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "T")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) != 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 1:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError(None)

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("XXParameter 't' cannot be emptyXX")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("PARAMETER 'T' CANNOT BE EMPTY")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(None):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr <= 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 1):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError(None)

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("XXTime values (t) must be non-negativeXX")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("TIME VALUES (T) MUST BE NON-NEGATIVE")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError(None)

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = None

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(None, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, None, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, None) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_32(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, ) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_33(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_34(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = None

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_35(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(None, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_36(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, None)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_37(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_38(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, )

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_39(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = None
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_40(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(None)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_41(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_42(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(None):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_43(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(None)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_44(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError(None)

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_45(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("XXPrediction resulted in non-finite valuesXX")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_46(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_47(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("PREDICTION RESULTED IN NON-FINITE VALUES")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_48(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = None

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_49(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(None, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_50(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, None):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_51(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_52(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, ):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_53(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_54(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(None):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_55(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(None)

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_56(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = None

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_57(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            None,
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_58(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(None, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_59(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, None, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_60(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, None, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_61(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, None, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_62(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, None) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_63(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_64(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_65(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_66(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_67(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, ) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_68(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(None, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_69(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, None)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_70(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_71(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, )],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_72(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if np.all(np.isfinite(rates)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_73(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(None):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_74(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(None)):
            raise ValueError("Derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_75(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError(None)

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_76(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("XXDerivative calculation resulted in non-finite valuesXX")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_77(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("derivative calculation resulted in non-finite values")

        return rates

    def xǁBassModelǁpredict_adoption_rate__mutmut_78(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        # Validate inputs
        t_arr = validate_sequence_numeric(t, "t")
        if len(t_arr) == 0:
            raise ValueError("Parameter 't' cannot be empty")

        # Validate that all time values are non-negative
        if np.any(t_arr < 0):
            raise ValueError("Time values (t) must be non-negative")

        # Validate model is fitted
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Validate covariates if provided
        validated_covariates = (
            validate_covariates_dict(covariates, self.covariates, len(t_arr)) if covariates is not None else None
        )

        y_pred = self.predict(t_arr, validated_covariates)

        # Validate that y_pred is finite
        y_pred = np.asarray(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction resulted in non-finite values")

        params = [self._params[name] for name in self.param_names]

        # Validate parameter values
        for param_name, param_val in zip(self.param_names, params):
            if not np.isfinite(param_val):
                raise ValueError(f"Parameter '{param_name}' must be finite, got {param_val}")

        rates = np.array(
            [self.differential_equation(ti, yi, params, validated_covariates, t_arr) for ti, yi in zip(t_arr, y_pred)],
        )

        if not np.all(np.isfinite(rates)):
            raise ValueError("DERIVATIVE CALCULATION RESULTED IN NON-FINITE VALUES")

        return rates
    
    xǁBassModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁpredict_adoption_rate__mutmut_1': xǁBassModelǁpredict_adoption_rate__mutmut_1, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_2': xǁBassModelǁpredict_adoption_rate__mutmut_2, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_3': xǁBassModelǁpredict_adoption_rate__mutmut_3, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_4': xǁBassModelǁpredict_adoption_rate__mutmut_4, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_5': xǁBassModelǁpredict_adoption_rate__mutmut_5, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_6': xǁBassModelǁpredict_adoption_rate__mutmut_6, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_7': xǁBassModelǁpredict_adoption_rate__mutmut_7, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_8': xǁBassModelǁpredict_adoption_rate__mutmut_8, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_9': xǁBassModelǁpredict_adoption_rate__mutmut_9, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_10': xǁBassModelǁpredict_adoption_rate__mutmut_10, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_11': xǁBassModelǁpredict_adoption_rate__mutmut_11, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_12': xǁBassModelǁpredict_adoption_rate__mutmut_12, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_13': xǁBassModelǁpredict_adoption_rate__mutmut_13, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_14': xǁBassModelǁpredict_adoption_rate__mutmut_14, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_15': xǁBassModelǁpredict_adoption_rate__mutmut_15, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_16': xǁBassModelǁpredict_adoption_rate__mutmut_16, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_17': xǁBassModelǁpredict_adoption_rate__mutmut_17, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_18': xǁBassModelǁpredict_adoption_rate__mutmut_18, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_19': xǁBassModelǁpredict_adoption_rate__mutmut_19, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_20': xǁBassModelǁpredict_adoption_rate__mutmut_20, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_21': xǁBassModelǁpredict_adoption_rate__mutmut_21, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_22': xǁBassModelǁpredict_adoption_rate__mutmut_22, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_23': xǁBassModelǁpredict_adoption_rate__mutmut_23, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_24': xǁBassModelǁpredict_adoption_rate__mutmut_24, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_25': xǁBassModelǁpredict_adoption_rate__mutmut_25, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_26': xǁBassModelǁpredict_adoption_rate__mutmut_26, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_27': xǁBassModelǁpredict_adoption_rate__mutmut_27, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_28': xǁBassModelǁpredict_adoption_rate__mutmut_28, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_29': xǁBassModelǁpredict_adoption_rate__mutmut_29, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_30': xǁBassModelǁpredict_adoption_rate__mutmut_30, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_31': xǁBassModelǁpredict_adoption_rate__mutmut_31, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_32': xǁBassModelǁpredict_adoption_rate__mutmut_32, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_33': xǁBassModelǁpredict_adoption_rate__mutmut_33, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_34': xǁBassModelǁpredict_adoption_rate__mutmut_34, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_35': xǁBassModelǁpredict_adoption_rate__mutmut_35, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_36': xǁBassModelǁpredict_adoption_rate__mutmut_36, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_37': xǁBassModelǁpredict_adoption_rate__mutmut_37, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_38': xǁBassModelǁpredict_adoption_rate__mutmut_38, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_39': xǁBassModelǁpredict_adoption_rate__mutmut_39, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_40': xǁBassModelǁpredict_adoption_rate__mutmut_40, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_41': xǁBassModelǁpredict_adoption_rate__mutmut_41, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_42': xǁBassModelǁpredict_adoption_rate__mutmut_42, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_43': xǁBassModelǁpredict_adoption_rate__mutmut_43, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_44': xǁBassModelǁpredict_adoption_rate__mutmut_44, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_45': xǁBassModelǁpredict_adoption_rate__mutmut_45, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_46': xǁBassModelǁpredict_adoption_rate__mutmut_46, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_47': xǁBassModelǁpredict_adoption_rate__mutmut_47, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_48': xǁBassModelǁpredict_adoption_rate__mutmut_48, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_49': xǁBassModelǁpredict_adoption_rate__mutmut_49, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_50': xǁBassModelǁpredict_adoption_rate__mutmut_50, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_51': xǁBassModelǁpredict_adoption_rate__mutmut_51, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_52': xǁBassModelǁpredict_adoption_rate__mutmut_52, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_53': xǁBassModelǁpredict_adoption_rate__mutmut_53, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_54': xǁBassModelǁpredict_adoption_rate__mutmut_54, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_55': xǁBassModelǁpredict_adoption_rate__mutmut_55, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_56': xǁBassModelǁpredict_adoption_rate__mutmut_56, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_57': xǁBassModelǁpredict_adoption_rate__mutmut_57, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_58': xǁBassModelǁpredict_adoption_rate__mutmut_58, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_59': xǁBassModelǁpredict_adoption_rate__mutmut_59, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_60': xǁBassModelǁpredict_adoption_rate__mutmut_60, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_61': xǁBassModelǁpredict_adoption_rate__mutmut_61, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_62': xǁBassModelǁpredict_adoption_rate__mutmut_62, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_63': xǁBassModelǁpredict_adoption_rate__mutmut_63, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_64': xǁBassModelǁpredict_adoption_rate__mutmut_64, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_65': xǁBassModelǁpredict_adoption_rate__mutmut_65, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_66': xǁBassModelǁpredict_adoption_rate__mutmut_66, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_67': xǁBassModelǁpredict_adoption_rate__mutmut_67, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_68': xǁBassModelǁpredict_adoption_rate__mutmut_68, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_69': xǁBassModelǁpredict_adoption_rate__mutmut_69, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_70': xǁBassModelǁpredict_adoption_rate__mutmut_70, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_71': xǁBassModelǁpredict_adoption_rate__mutmut_71, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_72': xǁBassModelǁpredict_adoption_rate__mutmut_72, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_73': xǁBassModelǁpredict_adoption_rate__mutmut_73, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_74': xǁBassModelǁpredict_adoption_rate__mutmut_74, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_75': xǁBassModelǁpredict_adoption_rate__mutmut_75, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_76': xǁBassModelǁpredict_adoption_rate__mutmut_76, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_77': xǁBassModelǁpredict_adoption_rate__mutmut_77, 
        'xǁBassModelǁpredict_adoption_rate__mutmut_78': xǁBassModelǁpredict_adoption_rate__mutmut_78
    }
    xǁBassModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁBassModelǁpredict_adoption_rate'

    def cumulative_adoption(self, t: Sequence[float], *params) -> np.ndarray:
        args = [t, *params]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBassModelǁcumulative_adoption__mutmut_orig'), object.__getattribute__(self, 'xǁBassModelǁcumulative_adoption__mutmut_mutants'), args, kwargs, self)

    def xǁBassModelǁcumulative_adoption__mutmut_orig(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_1(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = None
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_2(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(None)
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_3(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(None, params))
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_4(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(self.param_names, None))
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_5(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(params))
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_6(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(self.param_names, ))
        return self.predict(t)

    def xǁBassModelǁcumulative_adoption__mutmut_7(self, t: Sequence[float], *params) -> np.ndarray:
        self.params_ = dict(zip(self.param_names, params))
        return self.predict(None)
    
    xǁBassModelǁcumulative_adoption__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBassModelǁcumulative_adoption__mutmut_1': xǁBassModelǁcumulative_adoption__mutmut_1, 
        'xǁBassModelǁcumulative_adoption__mutmut_2': xǁBassModelǁcumulative_adoption__mutmut_2, 
        'xǁBassModelǁcumulative_adoption__mutmut_3': xǁBassModelǁcumulative_adoption__mutmut_3, 
        'xǁBassModelǁcumulative_adoption__mutmut_4': xǁBassModelǁcumulative_adoption__mutmut_4, 
        'xǁBassModelǁcumulative_adoption__mutmut_5': xǁBassModelǁcumulative_adoption__mutmut_5, 
        'xǁBassModelǁcumulative_adoption__mutmut_6': xǁBassModelǁcumulative_adoption__mutmut_6, 
        'xǁBassModelǁcumulative_adoption__mutmut_7': xǁBassModelǁcumulative_adoption__mutmut_7
    }
    xǁBassModelǁcumulative_adoption__mutmut_orig.__name__ = 'xǁBassModelǁcumulative_adoption'
