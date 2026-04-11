from collections.abc import Sequence

import numpy as np

from innovate.backend import current_backend as B
from innovate.base.base import DiffusionModel
from innovate.dynamics.growth.skewed import SkewedGrowth
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


class GompertzModel(DiffusionModel):
    """Implementation of the Gompertz Diffusion Model.
    This is a wrapper around the SkewedGrowth dynamics model.
    """

    def __init__(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        args = [covariates, t_event]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁ__init____mutmut_orig(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a Gompertz diffusion model with optional covariates.

        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = SkewedGrowth()

    def xǁGompertzModelǁ__init____mutmut_1(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a Gompertz diffusion model with optional covariates.

        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: dict[str, float] = None
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = SkewedGrowth()

    def xǁGompertzModelǁ__init____mutmut_2(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a Gompertz diffusion model with optional covariates.

        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: dict[str, float] = {}
        self.covariates = None
        self.t_event = t_event
        self.growth_model = SkewedGrowth()

    def xǁGompertzModelǁ__init____mutmut_3(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a Gompertz diffusion model with optional covariates.

        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates and []
        self.t_event = t_event
        self.growth_model = SkewedGrowth()

    def xǁGompertzModelǁ__init____mutmut_4(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a Gompertz diffusion model with optional covariates.

        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = None
        self.growth_model = SkewedGrowth()

    def xǁGompertzModelǁ__init____mutmut_5(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a Gompertz diffusion model with optional covariates.

        Creates an empty parameter dictionary, stores the provided covariate names, and instantiates a SkewedGrowth dynamics model for growth rate computation.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = None
    
    xǁGompertzModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁ__init____mutmut_1': xǁGompertzModelǁ__init____mutmut_1, 
        'xǁGompertzModelǁ__init____mutmut_2': xǁGompertzModelǁ__init____mutmut_2, 
        'xǁGompertzModelǁ__init____mutmut_3': xǁGompertzModelǁ__init____mutmut_3, 
        'xǁGompertzModelǁ__init____mutmut_4': xǁGompertzModelǁ__init____mutmut_4, 
        'xǁGompertzModelǁ__init____mutmut_5': xǁGompertzModelǁ__init____mutmut_5
    }
    xǁGompertzModelǁ__init____mutmut_orig.__name__ = 'xǁGompertzModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        """Return the list of model parameter names, including base parameters and covariate-specific coefficients.

        Returns
        -------
            Sequence[str]: List of parameter names for the model, with additional parameters for each covariate in the form 'beta_a_{cov}', 'beta_b_{cov}', and 'beta_c_{cov}'.
        """
        names = ["a", "b", "c"]
        if self.t_event is not None:
            names.extend(["a_post", "b_post", "c_post"])
        for cov in self.covariates:
            names.extend([f"beta_a_{cov}", f"beta_b_{cov}", f"beta_c_{cov}"])
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = None
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "XXaXX": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "A": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) / 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(None) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 2.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "XXbXX": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "B": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 2.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "XXcXX": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "C": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 1.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                None,
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "XXa_postXX": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "A_POST": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) / 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(None) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 2.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "XXb_postXX": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "B_POST": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 2.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "XXc_postXX": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "C_POST": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 1.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = None
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 1.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = None
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 1.0
            guesses[f"beta_c_{cov}"] = 0.0
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = None
        return guesses

    def xǁGompertzModelǁinitial_guesses__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "a": np.max(y) * 1.1,
            "b": 1.0,
            "c": 0.1,
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "a_post": np.max(y) * 1.1,
                    "b_post": 1.0,
                    "c_post": 0.1,
                },
            )
        for cov in self.covariates:
            guesses[f"beta_a_{cov}"] = 0.0
            guesses[f"beta_b_{cov}"] = 0.0
            guesses[f"beta_c_{cov}"] = 1.0
        return guesses
    
    xǁGompertzModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁinitial_guesses__mutmut_1': xǁGompertzModelǁinitial_guesses__mutmut_1, 
        'xǁGompertzModelǁinitial_guesses__mutmut_2': xǁGompertzModelǁinitial_guesses__mutmut_2, 
        'xǁGompertzModelǁinitial_guesses__mutmut_3': xǁGompertzModelǁinitial_guesses__mutmut_3, 
        'xǁGompertzModelǁinitial_guesses__mutmut_4': xǁGompertzModelǁinitial_guesses__mutmut_4, 
        'xǁGompertzModelǁinitial_guesses__mutmut_5': xǁGompertzModelǁinitial_guesses__mutmut_5, 
        'xǁGompertzModelǁinitial_guesses__mutmut_6': xǁGompertzModelǁinitial_guesses__mutmut_6, 
        'xǁGompertzModelǁinitial_guesses__mutmut_7': xǁGompertzModelǁinitial_guesses__mutmut_7, 
        'xǁGompertzModelǁinitial_guesses__mutmut_8': xǁGompertzModelǁinitial_guesses__mutmut_8, 
        'xǁGompertzModelǁinitial_guesses__mutmut_9': xǁGompertzModelǁinitial_guesses__mutmut_9, 
        'xǁGompertzModelǁinitial_guesses__mutmut_10': xǁGompertzModelǁinitial_guesses__mutmut_10, 
        'xǁGompertzModelǁinitial_guesses__mutmut_11': xǁGompertzModelǁinitial_guesses__mutmut_11, 
        'xǁGompertzModelǁinitial_guesses__mutmut_12': xǁGompertzModelǁinitial_guesses__mutmut_12, 
        'xǁGompertzModelǁinitial_guesses__mutmut_13': xǁGompertzModelǁinitial_guesses__mutmut_13, 
        'xǁGompertzModelǁinitial_guesses__mutmut_14': xǁGompertzModelǁinitial_guesses__mutmut_14, 
        'xǁGompertzModelǁinitial_guesses__mutmut_15': xǁGompertzModelǁinitial_guesses__mutmut_15, 
        'xǁGompertzModelǁinitial_guesses__mutmut_16': xǁGompertzModelǁinitial_guesses__mutmut_16, 
        'xǁGompertzModelǁinitial_guesses__mutmut_17': xǁGompertzModelǁinitial_guesses__mutmut_17, 
        'xǁGompertzModelǁinitial_guesses__mutmut_18': xǁGompertzModelǁinitial_guesses__mutmut_18, 
        'xǁGompertzModelǁinitial_guesses__mutmut_19': xǁGompertzModelǁinitial_guesses__mutmut_19, 
        'xǁGompertzModelǁinitial_guesses__mutmut_20': xǁGompertzModelǁinitial_guesses__mutmut_20, 
        'xǁGompertzModelǁinitial_guesses__mutmut_21': xǁGompertzModelǁinitial_guesses__mutmut_21, 
        'xǁGompertzModelǁinitial_guesses__mutmut_22': xǁGompertzModelǁinitial_guesses__mutmut_22, 
        'xǁGompertzModelǁinitial_guesses__mutmut_23': xǁGompertzModelǁinitial_guesses__mutmut_23, 
        'xǁGompertzModelǁinitial_guesses__mutmut_24': xǁGompertzModelǁinitial_guesses__mutmut_24, 
        'xǁGompertzModelǁinitial_guesses__mutmut_25': xǁGompertzModelǁinitial_guesses__mutmut_25, 
        'xǁGompertzModelǁinitial_guesses__mutmut_26': xǁGompertzModelǁinitial_guesses__mutmut_26, 
        'xǁGompertzModelǁinitial_guesses__mutmut_27': xǁGompertzModelǁinitial_guesses__mutmut_27, 
        'xǁGompertzModelǁinitial_guesses__mutmut_28': xǁGompertzModelǁinitial_guesses__mutmut_28, 
        'xǁGompertzModelǁinitial_guesses__mutmut_29': xǁGompertzModelǁinitial_guesses__mutmut_29, 
        'xǁGompertzModelǁinitial_guesses__mutmut_30': xǁGompertzModelǁinitial_guesses__mutmut_30, 
        'xǁGompertzModelǁinitial_guesses__mutmut_31': xǁGompertzModelǁinitial_guesses__mutmut_31
    }
    xǁGompertzModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁGompertzModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = None
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "XXaXX": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "A": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(None), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "XXbXX": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "B": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1.000001, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "XXcXX": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "C": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1.000001, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                None,
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "XXa_postXX": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "A_POST": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(None), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "XXb_postXX": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "B_POST": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1.000001, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "XXc_postXX": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "C_POST": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1.000001, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = None
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (+np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = None
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (+np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁGompertzModelǁbounds__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = None
        return bounds

    def xǁGompertzModelǁbounds__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the Gompertz model based on observed data and covariates.

        The bounds ensure that the main parameters are constrained to meaningful ranges, while covariate effect parameters are unbounded.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observed data.
            y (Sequence[float]): Observed cumulative adoption values.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to (lower, upper) bounds.
        """
        bounds = {
            "a": (np.max(y), np.inf),
            "b": (1e-6, np.inf),
            "c": (1e-6, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "a_post": (np.max(y), np.inf),
                    "b_post": (1e-6, np.inf),
                    "c_post": (1e-6, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_a_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_b_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_c_{cov}"] = (+np.inf, np.inf)
        return bounds
    
    xǁGompertzModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁbounds__mutmut_1': xǁGompertzModelǁbounds__mutmut_1, 
        'xǁGompertzModelǁbounds__mutmut_2': xǁGompertzModelǁbounds__mutmut_2, 
        'xǁGompertzModelǁbounds__mutmut_3': xǁGompertzModelǁbounds__mutmut_3, 
        'xǁGompertzModelǁbounds__mutmut_4': xǁGompertzModelǁbounds__mutmut_4, 
        'xǁGompertzModelǁbounds__mutmut_5': xǁGompertzModelǁbounds__mutmut_5, 
        'xǁGompertzModelǁbounds__mutmut_6': xǁGompertzModelǁbounds__mutmut_6, 
        'xǁGompertzModelǁbounds__mutmut_7': xǁGompertzModelǁbounds__mutmut_7, 
        'xǁGompertzModelǁbounds__mutmut_8': xǁGompertzModelǁbounds__mutmut_8, 
        'xǁGompertzModelǁbounds__mutmut_9': xǁGompertzModelǁbounds__mutmut_9, 
        'xǁGompertzModelǁbounds__mutmut_10': xǁGompertzModelǁbounds__mutmut_10, 
        'xǁGompertzModelǁbounds__mutmut_11': xǁGompertzModelǁbounds__mutmut_11, 
        'xǁGompertzModelǁbounds__mutmut_12': xǁGompertzModelǁbounds__mutmut_12, 
        'xǁGompertzModelǁbounds__mutmut_13': xǁGompertzModelǁbounds__mutmut_13, 
        'xǁGompertzModelǁbounds__mutmut_14': xǁGompertzModelǁbounds__mutmut_14, 
        'xǁGompertzModelǁbounds__mutmut_15': xǁGompertzModelǁbounds__mutmut_15, 
        'xǁGompertzModelǁbounds__mutmut_16': xǁGompertzModelǁbounds__mutmut_16, 
        'xǁGompertzModelǁbounds__mutmut_17': xǁGompertzModelǁbounds__mutmut_17, 
        'xǁGompertzModelǁbounds__mutmut_18': xǁGompertzModelǁbounds__mutmut_18, 
        'xǁGompertzModelǁbounds__mutmut_19': xǁGompertzModelǁbounds__mutmut_19, 
        'xǁGompertzModelǁbounds__mutmut_20': xǁGompertzModelǁbounds__mutmut_20, 
        'xǁGompertzModelǁbounds__mutmut_21': xǁGompertzModelǁbounds__mutmut_21, 
        'xǁGompertzModelǁbounds__mutmut_22': xǁGompertzModelǁbounds__mutmut_22, 
        'xǁGompertzModelǁbounds__mutmut_23': xǁGompertzModelǁbounds__mutmut_23, 
        'xǁGompertzModelǁbounds__mutmut_24': xǁGompertzModelǁbounds__mutmut_24, 
        'xǁGompertzModelǁbounds__mutmut_25': xǁGompertzModelǁbounds__mutmut_25, 
        'xǁGompertzModelǁbounds__mutmut_26': xǁGompertzModelǁbounds__mutmut_26, 
        'xǁGompertzModelǁbounds__mutmut_27': xǁGompertzModelǁbounds__mutmut_27
    }
    xǁGompertzModelǁbounds__mutmut_orig.__name__ = 'xǁGompertzModelǁbounds'

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁpredict__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError(None)

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = None

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(None, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, None, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, None, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, None, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, None)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, )

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = None
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1.000001]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = None
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            None,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            None,
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            None,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=None,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method=None,
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=None,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_32(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[1], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_33(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[+1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_34(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-2]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_35(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="XXLSODAXX",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_36(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="lsoda",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_37(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=False,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_38(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = None
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_39(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(None).flatten()
        return np.maximum.accumulate(y_pred)

    def xǁGompertzModelǁpredict__mutmut_40(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts cumulative adoption values at specified times using the fitted Gompertz diffusion model.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to predict cumulative adoption.
            covariates (Dict[str, Sequence[float]], optional): Time series of covariate values affecting the model parameters.

        Returns
        -------
            Sequence[float]: Predicted cumulative adoption values at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set via fitting.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # This is a simplification. The predict method should use the growth model's
        # predict_cumulative method, which will require some refactoring of how parameters
        # are handled. For now, we will leave the old implementation.
        from scipy.integrate import solve_ivp

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        y0 = [1e-6]
        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
            dense_output=True,
        )
        y_pred = sol.sol(t).flatten()
        return np.maximum.accumulate(None)
    
    xǁGompertzModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁpredict__mutmut_1': xǁGompertzModelǁpredict__mutmut_1, 
        'xǁGompertzModelǁpredict__mutmut_2': xǁGompertzModelǁpredict__mutmut_2, 
        'xǁGompertzModelǁpredict__mutmut_3': xǁGompertzModelǁpredict__mutmut_3, 
        'xǁGompertzModelǁpredict__mutmut_4': xǁGompertzModelǁpredict__mutmut_4, 
        'xǁGompertzModelǁpredict__mutmut_5': xǁGompertzModelǁpredict__mutmut_5, 
        'xǁGompertzModelǁpredict__mutmut_6': xǁGompertzModelǁpredict__mutmut_6, 
        'xǁGompertzModelǁpredict__mutmut_7': xǁGompertzModelǁpredict__mutmut_7, 
        'xǁGompertzModelǁpredict__mutmut_8': xǁGompertzModelǁpredict__mutmut_8, 
        'xǁGompertzModelǁpredict__mutmut_9': xǁGompertzModelǁpredict__mutmut_9, 
        'xǁGompertzModelǁpredict__mutmut_10': xǁGompertzModelǁpredict__mutmut_10, 
        'xǁGompertzModelǁpredict__mutmut_11': xǁGompertzModelǁpredict__mutmut_11, 
        'xǁGompertzModelǁpredict__mutmut_12': xǁGompertzModelǁpredict__mutmut_12, 
        'xǁGompertzModelǁpredict__mutmut_13': xǁGompertzModelǁpredict__mutmut_13, 
        'xǁGompertzModelǁpredict__mutmut_14': xǁGompertzModelǁpredict__mutmut_14, 
        'xǁGompertzModelǁpredict__mutmut_15': xǁGompertzModelǁpredict__mutmut_15, 
        'xǁGompertzModelǁpredict__mutmut_16': xǁGompertzModelǁpredict__mutmut_16, 
        'xǁGompertzModelǁpredict__mutmut_17': xǁGompertzModelǁpredict__mutmut_17, 
        'xǁGompertzModelǁpredict__mutmut_18': xǁGompertzModelǁpredict__mutmut_18, 
        'xǁGompertzModelǁpredict__mutmut_19': xǁGompertzModelǁpredict__mutmut_19, 
        'xǁGompertzModelǁpredict__mutmut_20': xǁGompertzModelǁpredict__mutmut_20, 
        'xǁGompertzModelǁpredict__mutmut_21': xǁGompertzModelǁpredict__mutmut_21, 
        'xǁGompertzModelǁpredict__mutmut_22': xǁGompertzModelǁpredict__mutmut_22, 
        'xǁGompertzModelǁpredict__mutmut_23': xǁGompertzModelǁpredict__mutmut_23, 
        'xǁGompertzModelǁpredict__mutmut_24': xǁGompertzModelǁpredict__mutmut_24, 
        'xǁGompertzModelǁpredict__mutmut_25': xǁGompertzModelǁpredict__mutmut_25, 
        'xǁGompertzModelǁpredict__mutmut_26': xǁGompertzModelǁpredict__mutmut_26, 
        'xǁGompertzModelǁpredict__mutmut_27': xǁGompertzModelǁpredict__mutmut_27, 
        'xǁGompertzModelǁpredict__mutmut_28': xǁGompertzModelǁpredict__mutmut_28, 
        'xǁGompertzModelǁpredict__mutmut_29': xǁGompertzModelǁpredict__mutmut_29, 
        'xǁGompertzModelǁpredict__mutmut_30': xǁGompertzModelǁpredict__mutmut_30, 
        'xǁGompertzModelǁpredict__mutmut_31': xǁGompertzModelǁpredict__mutmut_31, 
        'xǁGompertzModelǁpredict__mutmut_32': xǁGompertzModelǁpredict__mutmut_32, 
        'xǁGompertzModelǁpredict__mutmut_33': xǁGompertzModelǁpredict__mutmut_33, 
        'xǁGompertzModelǁpredict__mutmut_34': xǁGompertzModelǁpredict__mutmut_34, 
        'xǁGompertzModelǁpredict__mutmut_35': xǁGompertzModelǁpredict__mutmut_35, 
        'xǁGompertzModelǁpredict__mutmut_36': xǁGompertzModelǁpredict__mutmut_36, 
        'xǁGompertzModelǁpredict__mutmut_37': xǁGompertzModelǁpredict__mutmut_37, 
        'xǁGompertzModelǁpredict__mutmut_38': xǁGompertzModelǁpredict__mutmut_38, 
        'xǁGompertzModelǁpredict__mutmut_39': xǁGompertzModelǁpredict__mutmut_39, 
        'xǁGompertzModelǁpredict__mutmut_40': xǁGompertzModelǁpredict__mutmut_40
    }
    xǁGompertzModelǁpredict__mutmut_orig.__name__ = 'xǁGompertzModelǁpredict'

    def differential_equation(self, t, y, params, covariates, t_eval):
        args = [t, y, params, covariates, t_eval]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁdifferential_equation__mutmut_orig(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_1(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None or t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_2(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_3(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t > self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_4(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = None
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_5(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[4]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_6(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = None
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_7(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[5]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_8(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = None
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_9(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[6]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_10(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = None
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_11(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 4
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_12(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = None
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_13(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[1]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_14(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = None
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_15(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[2]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_16(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = None
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_17(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[3]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_18(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = None

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_19(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 1

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_20(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = None
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_21(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = None
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_22(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = None

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_23(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = None
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_24(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 - param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_25(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 4 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_26(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = None

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_27(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(None, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_28(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, None, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_29(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, None)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_30(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_31(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_32(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, )

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_33(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t = params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_34(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t -= params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_35(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] / cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_36(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t = params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_37(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t -= params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_38(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] / cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_39(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx - 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_40(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 2] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_41(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t = params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_42(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t -= params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_43(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] / cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_44(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx - 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_45(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 3] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_46(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx = 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_47(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx -= 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_48(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 4

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_49(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            None,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_50(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            None,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_51(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=None,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_52(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=None,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_53(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=None,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_54(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            a_t,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_55(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            t=t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_56(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            shape_b=b_t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_57(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_c=c_t,
        )

    def xǁGompertzModelǁdifferential_equation__mutmut_58(self, t, y, params, covariates, t_eval):
        """Defines the time derivative for the Gompertz diffusion model, incorporating covariate effects by adjusting parameters at time t.

        Parameters
        ----------
            t (float): Current time point.
            y (float): Current cumulative adoption value.
            params (Sequence[float]): Model parameters, including base and covariate coefficients.
            covariates (dict or None): Optional mapping of covariate names to their time series values.
            t_eval (Sequence[float]): Time points corresponding to covariate values.

        Returns
        -------
            float: The instantaneous growth rate at time t.
        """
        if self.t_event is not None and t >= self.t_event:
            a_base = params[3]
            b_base = params[4]
            c_base = params[5]
            param_idx_offset = 3
        else:
            a_base = params[0]
            b_base = params[1]
            c_base = params[2]
            param_idx_offset = 0

        a_t = a_base
        b_t = b_base
        c_t = c_base

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)

                a_t += params[param_idx] * cov_val_t
                b_t += params[param_idx + 1] * cov_val_t
                c_t += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return self.growth_model.compute_growth_rate(
            y,
            a_t,
            t=t,
            shape_b=b_t,
            )
    
    xǁGompertzModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁdifferential_equation__mutmut_1': xǁGompertzModelǁdifferential_equation__mutmut_1, 
        'xǁGompertzModelǁdifferential_equation__mutmut_2': xǁGompertzModelǁdifferential_equation__mutmut_2, 
        'xǁGompertzModelǁdifferential_equation__mutmut_3': xǁGompertzModelǁdifferential_equation__mutmut_3, 
        'xǁGompertzModelǁdifferential_equation__mutmut_4': xǁGompertzModelǁdifferential_equation__mutmut_4, 
        'xǁGompertzModelǁdifferential_equation__mutmut_5': xǁGompertzModelǁdifferential_equation__mutmut_5, 
        'xǁGompertzModelǁdifferential_equation__mutmut_6': xǁGompertzModelǁdifferential_equation__mutmut_6, 
        'xǁGompertzModelǁdifferential_equation__mutmut_7': xǁGompertzModelǁdifferential_equation__mutmut_7, 
        'xǁGompertzModelǁdifferential_equation__mutmut_8': xǁGompertzModelǁdifferential_equation__mutmut_8, 
        'xǁGompertzModelǁdifferential_equation__mutmut_9': xǁGompertzModelǁdifferential_equation__mutmut_9, 
        'xǁGompertzModelǁdifferential_equation__mutmut_10': xǁGompertzModelǁdifferential_equation__mutmut_10, 
        'xǁGompertzModelǁdifferential_equation__mutmut_11': xǁGompertzModelǁdifferential_equation__mutmut_11, 
        'xǁGompertzModelǁdifferential_equation__mutmut_12': xǁGompertzModelǁdifferential_equation__mutmut_12, 
        'xǁGompertzModelǁdifferential_equation__mutmut_13': xǁGompertzModelǁdifferential_equation__mutmut_13, 
        'xǁGompertzModelǁdifferential_equation__mutmut_14': xǁGompertzModelǁdifferential_equation__mutmut_14, 
        'xǁGompertzModelǁdifferential_equation__mutmut_15': xǁGompertzModelǁdifferential_equation__mutmut_15, 
        'xǁGompertzModelǁdifferential_equation__mutmut_16': xǁGompertzModelǁdifferential_equation__mutmut_16, 
        'xǁGompertzModelǁdifferential_equation__mutmut_17': xǁGompertzModelǁdifferential_equation__mutmut_17, 
        'xǁGompertzModelǁdifferential_equation__mutmut_18': xǁGompertzModelǁdifferential_equation__mutmut_18, 
        'xǁGompertzModelǁdifferential_equation__mutmut_19': xǁGompertzModelǁdifferential_equation__mutmut_19, 
        'xǁGompertzModelǁdifferential_equation__mutmut_20': xǁGompertzModelǁdifferential_equation__mutmut_20, 
        'xǁGompertzModelǁdifferential_equation__mutmut_21': xǁGompertzModelǁdifferential_equation__mutmut_21, 
        'xǁGompertzModelǁdifferential_equation__mutmut_22': xǁGompertzModelǁdifferential_equation__mutmut_22, 
        'xǁGompertzModelǁdifferential_equation__mutmut_23': xǁGompertzModelǁdifferential_equation__mutmut_23, 
        'xǁGompertzModelǁdifferential_equation__mutmut_24': xǁGompertzModelǁdifferential_equation__mutmut_24, 
        'xǁGompertzModelǁdifferential_equation__mutmut_25': xǁGompertzModelǁdifferential_equation__mutmut_25, 
        'xǁGompertzModelǁdifferential_equation__mutmut_26': xǁGompertzModelǁdifferential_equation__mutmut_26, 
        'xǁGompertzModelǁdifferential_equation__mutmut_27': xǁGompertzModelǁdifferential_equation__mutmut_27, 
        'xǁGompertzModelǁdifferential_equation__mutmut_28': xǁGompertzModelǁdifferential_equation__mutmut_28, 
        'xǁGompertzModelǁdifferential_equation__mutmut_29': xǁGompertzModelǁdifferential_equation__mutmut_29, 
        'xǁGompertzModelǁdifferential_equation__mutmut_30': xǁGompertzModelǁdifferential_equation__mutmut_30, 
        'xǁGompertzModelǁdifferential_equation__mutmut_31': xǁGompertzModelǁdifferential_equation__mutmut_31, 
        'xǁGompertzModelǁdifferential_equation__mutmut_32': xǁGompertzModelǁdifferential_equation__mutmut_32, 
        'xǁGompertzModelǁdifferential_equation__mutmut_33': xǁGompertzModelǁdifferential_equation__mutmut_33, 
        'xǁGompertzModelǁdifferential_equation__mutmut_34': xǁGompertzModelǁdifferential_equation__mutmut_34, 
        'xǁGompertzModelǁdifferential_equation__mutmut_35': xǁGompertzModelǁdifferential_equation__mutmut_35, 
        'xǁGompertzModelǁdifferential_equation__mutmut_36': xǁGompertzModelǁdifferential_equation__mutmut_36, 
        'xǁGompertzModelǁdifferential_equation__mutmut_37': xǁGompertzModelǁdifferential_equation__mutmut_37, 
        'xǁGompertzModelǁdifferential_equation__mutmut_38': xǁGompertzModelǁdifferential_equation__mutmut_38, 
        'xǁGompertzModelǁdifferential_equation__mutmut_39': xǁGompertzModelǁdifferential_equation__mutmut_39, 
        'xǁGompertzModelǁdifferential_equation__mutmut_40': xǁGompertzModelǁdifferential_equation__mutmut_40, 
        'xǁGompertzModelǁdifferential_equation__mutmut_41': xǁGompertzModelǁdifferential_equation__mutmut_41, 
        'xǁGompertzModelǁdifferential_equation__mutmut_42': xǁGompertzModelǁdifferential_equation__mutmut_42, 
        'xǁGompertzModelǁdifferential_equation__mutmut_43': xǁGompertzModelǁdifferential_equation__mutmut_43, 
        'xǁGompertzModelǁdifferential_equation__mutmut_44': xǁGompertzModelǁdifferential_equation__mutmut_44, 
        'xǁGompertzModelǁdifferential_equation__mutmut_45': xǁGompertzModelǁdifferential_equation__mutmut_45, 
        'xǁGompertzModelǁdifferential_equation__mutmut_46': xǁGompertzModelǁdifferential_equation__mutmut_46, 
        'xǁGompertzModelǁdifferential_equation__mutmut_47': xǁGompertzModelǁdifferential_equation__mutmut_47, 
        'xǁGompertzModelǁdifferential_equation__mutmut_48': xǁGompertzModelǁdifferential_equation__mutmut_48, 
        'xǁGompertzModelǁdifferential_equation__mutmut_49': xǁGompertzModelǁdifferential_equation__mutmut_49, 
        'xǁGompertzModelǁdifferential_equation__mutmut_50': xǁGompertzModelǁdifferential_equation__mutmut_50, 
        'xǁGompertzModelǁdifferential_equation__mutmut_51': xǁGompertzModelǁdifferential_equation__mutmut_51, 
        'xǁGompertzModelǁdifferential_equation__mutmut_52': xǁGompertzModelǁdifferential_equation__mutmut_52, 
        'xǁGompertzModelǁdifferential_equation__mutmut_53': xǁGompertzModelǁdifferential_equation__mutmut_53, 
        'xǁGompertzModelǁdifferential_equation__mutmut_54': xǁGompertzModelǁdifferential_equation__mutmut_54, 
        'xǁGompertzModelǁdifferential_equation__mutmut_55': xǁGompertzModelǁdifferential_equation__mutmut_55, 
        'xǁGompertzModelǁdifferential_equation__mutmut_56': xǁGompertzModelǁdifferential_equation__mutmut_56, 
        'xǁGompertzModelǁdifferential_equation__mutmut_57': xǁGompertzModelǁdifferential_equation__mutmut_57, 
        'xǁGompertzModelǁdifferential_equation__mutmut_58': xǁGompertzModelǁdifferential_equation__mutmut_58
    }
    xǁGompertzModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁGompertzModelǁdifferential_equation'

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        args = [t, y, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁscore__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = None
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(None, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, None)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, )
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = None
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum(None)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) * 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) + y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(None) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 3)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) + B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(None) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁGompertzModelǁscore__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁGompertzModelǁscore__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed data and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations are made.
            y (Sequence[float]): Observed cumulative adoption values.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model parameters have not been set.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁGompertzModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁscore__mutmut_1': xǁGompertzModelǁscore__mutmut_1, 
        'xǁGompertzModelǁscore__mutmut_2': xǁGompertzModelǁscore__mutmut_2, 
        'xǁGompertzModelǁscore__mutmut_3': xǁGompertzModelǁscore__mutmut_3, 
        'xǁGompertzModelǁscore__mutmut_4': xǁGompertzModelǁscore__mutmut_4, 
        'xǁGompertzModelǁscore__mutmut_5': xǁGompertzModelǁscore__mutmut_5, 
        'xǁGompertzModelǁscore__mutmut_6': xǁGompertzModelǁscore__mutmut_6, 
        'xǁGompertzModelǁscore__mutmut_7': xǁGompertzModelǁscore__mutmut_7, 
        'xǁGompertzModelǁscore__mutmut_8': xǁGompertzModelǁscore__mutmut_8, 
        'xǁGompertzModelǁscore__mutmut_9': xǁGompertzModelǁscore__mutmut_9, 
        'xǁGompertzModelǁscore__mutmut_10': xǁGompertzModelǁscore__mutmut_10, 
        'xǁGompertzModelǁscore__mutmut_11': xǁGompertzModelǁscore__mutmut_11, 
        'xǁGompertzModelǁscore__mutmut_12': xǁGompertzModelǁscore__mutmut_12, 
        'xǁGompertzModelǁscore__mutmut_13': xǁGompertzModelǁscore__mutmut_13, 
        'xǁGompertzModelǁscore__mutmut_14': xǁGompertzModelǁscore__mutmut_14, 
        'xǁGompertzModelǁscore__mutmut_15': xǁGompertzModelǁscore__mutmut_15, 
        'xǁGompertzModelǁscore__mutmut_16': xǁGompertzModelǁscore__mutmut_16, 
        'xǁGompertzModelǁscore__mutmut_17': xǁGompertzModelǁscore__mutmut_17, 
        'xǁGompertzModelǁscore__mutmut_18': xǁGompertzModelǁscore__mutmut_18, 
        'xǁGompertzModelǁscore__mutmut_19': xǁGompertzModelǁscore__mutmut_19, 
        'xǁGompertzModelǁscore__mutmut_20': xǁGompertzModelǁscore__mutmut_20, 
        'xǁGompertzModelǁscore__mutmut_21': xǁGompertzModelǁscore__mutmut_21, 
        'xǁGompertzModelǁscore__mutmut_22': xǁGompertzModelǁscore__mutmut_22, 
        'xǁGompertzModelǁscore__mutmut_23': xǁGompertzModelǁscore__mutmut_23, 
        'xǁGompertzModelǁscore__mutmut_24': xǁGompertzModelǁscore__mutmut_24, 
        'xǁGompertzModelǁscore__mutmut_25': xǁGompertzModelǁscore__mutmut_25, 
        'xǁGompertzModelǁscore__mutmut_26': xǁGompertzModelǁscore__mutmut_26, 
        'xǁGompertzModelǁscore__mutmut_27': xǁGompertzModelǁscore__mutmut_27, 
        'xǁGompertzModelǁscore__mutmut_28': xǁGompertzModelǁscore__mutmut_28, 
        'xǁGompertzModelǁscore__mutmut_29': xǁGompertzModelǁscore__mutmut_29
    }
    xǁGompertzModelǁscore__mutmut_orig.__name__ = 'xǁGompertzModelǁscore'

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
    ) -> Sequence[float]:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError(None)

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = None
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(None, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, None)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, )
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = None

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = None
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            None,
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(None, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, None, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, None, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, None, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, None) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, ) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(None, y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, None)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(y_pred)],
        )
        return rates

    def xǁGompertzModelǁpredict_adoption_rate__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = np.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, )],
        )
        return rates
    
    xǁGompertzModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁpredict_adoption_rate__mutmut_1': xǁGompertzModelǁpredict_adoption_rate__mutmut_1, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_2': xǁGompertzModelǁpredict_adoption_rate__mutmut_2, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_3': xǁGompertzModelǁpredict_adoption_rate__mutmut_3, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_4': xǁGompertzModelǁpredict_adoption_rate__mutmut_4, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_5': xǁGompertzModelǁpredict_adoption_rate__mutmut_5, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_6': xǁGompertzModelǁpredict_adoption_rate__mutmut_6, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_7': xǁGompertzModelǁpredict_adoption_rate__mutmut_7, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_8': xǁGompertzModelǁpredict_adoption_rate__mutmut_8, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_9': xǁGompertzModelǁpredict_adoption_rate__mutmut_9, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_10': xǁGompertzModelǁpredict_adoption_rate__mutmut_10, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_11': xǁGompertzModelǁpredict_adoption_rate__mutmut_11, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_12': xǁGompertzModelǁpredict_adoption_rate__mutmut_12, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_13': xǁGompertzModelǁpredict_adoption_rate__mutmut_13, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_14': xǁGompertzModelǁpredict_adoption_rate__mutmut_14, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_15': xǁGompertzModelǁpredict_adoption_rate__mutmut_15, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_16': xǁGompertzModelǁpredict_adoption_rate__mutmut_16, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_17': xǁGompertzModelǁpredict_adoption_rate__mutmut_17, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_18': xǁGompertzModelǁpredict_adoption_rate__mutmut_18, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_19': xǁGompertzModelǁpredict_adoption_rate__mutmut_19, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_20': xǁGompertzModelǁpredict_adoption_rate__mutmut_20, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_21': xǁGompertzModelǁpredict_adoption_rate__mutmut_21, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_22': xǁGompertzModelǁpredict_adoption_rate__mutmut_22, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_23': xǁGompertzModelǁpredict_adoption_rate__mutmut_23, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_24': xǁGompertzModelǁpredict_adoption_rate__mutmut_24, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_25': xǁGompertzModelǁpredict_adoption_rate__mutmut_25, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_26': xǁGompertzModelǁpredict_adoption_rate__mutmut_26, 
        'xǁGompertzModelǁpredict_adoption_rate__mutmut_27': xǁGompertzModelǁpredict_adoption_rate__mutmut_27
    }
    xǁGompertzModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁGompertzModelǁpredict_adoption_rate'

    def cumulative_adoption(self, t: Sequence[float], *params) -> Sequence[float]:
        args = [t, *params]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁGompertzModelǁcumulative_adoption__mutmut_orig'), object.__getattribute__(self, 'xǁGompertzModelǁcumulative_adoption__mutmut_mutants'), args, kwargs, self)

    def xǁGompertzModelǁcumulative_adoption__mutmut_orig(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_1(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = None
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_2(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(None)
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_3(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(None, params))
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_4(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(self.param_names, None))
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_5(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(params))
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_6(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(self.param_names, ))
        return self.predict(t)

    def xǁGompertzModelǁcumulative_adoption__mutmut_7(self, t: Sequence[float], *params) -> Sequence[float]:
        self.params_ = dict(zip(self.param_names, params))
        return self.predict(None)
    
    xǁGompertzModelǁcumulative_adoption__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁGompertzModelǁcumulative_adoption__mutmut_1': xǁGompertzModelǁcumulative_adoption__mutmut_1, 
        'xǁGompertzModelǁcumulative_adoption__mutmut_2': xǁGompertzModelǁcumulative_adoption__mutmut_2, 
        'xǁGompertzModelǁcumulative_adoption__mutmut_3': xǁGompertzModelǁcumulative_adoption__mutmut_3, 
        'xǁGompertzModelǁcumulative_adoption__mutmut_4': xǁGompertzModelǁcumulative_adoption__mutmut_4, 
        'xǁGompertzModelǁcumulative_adoption__mutmut_5': xǁGompertzModelǁcumulative_adoption__mutmut_5, 
        'xǁGompertzModelǁcumulative_adoption__mutmut_6': xǁGompertzModelǁcumulative_adoption__mutmut_6, 
        'xǁGompertzModelǁcumulative_adoption__mutmut_7': xǁGompertzModelǁcumulative_adoption__mutmut_7
    }
    xǁGompertzModelǁcumulative_adoption__mutmut_orig.__name__ = 'xǁGompertzModelǁcumulative_adoption'
