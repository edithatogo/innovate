from collections.abc import Sequence

import numpy as np

from innovate import backend
from innovate.base.base import DiffusionModel
from innovate.dynamics.growth.symmetric import SymmetricGrowth
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


class LogisticModel(DiffusionModel):
    """Implementation of the Logistic Diffusion Model.
    This is a wrapper around the SymmetricGrowth dynamics model.
    """

    def __init__(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        args = [covariates, t_event]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁ__init____mutmut_orig(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = SymmetricGrowth()

    def xǁLogisticModelǁ__init____mutmut_1(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: dict[str, float] = None
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = SymmetricGrowth()

    def xǁLogisticModelǁ__init____mutmut_2(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: dict[str, float] = {}
        self.covariates = None
        self.t_event = t_event
        self.growth_model = SymmetricGrowth()

    def xǁLogisticModelǁ__init____mutmut_3(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates and []
        self.t_event = t_event
        self.growth_model = SymmetricGrowth()

    def xǁLogisticModelǁ__init____mutmut_4(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = None
        self.growth_model = SymmetricGrowth()

    def xǁLogisticModelǁ__init____mutmut_5(
        self,
        covariates: Sequence[str] | None = None,
        t_event: float | None = None,
    ):
        """Initialize a LogisticModel with optional covariates and an internal SymmetricGrowth dynamics model.

        Parameters
        ----------
            covariates (Sequence[str], optional): List of covariate names to include in the model. Defaults to an empty list.
            t_event (float, optional): The time of a structural break or event.
        """
        self._params: dict[str, float] = {}
        self.covariates = covariates or []
        self.t_event = t_event
        self.growth_model = None
    
    xǁLogisticModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁ__init____mutmut_1': xǁLogisticModelǁ__init____mutmut_1, 
        'xǁLogisticModelǁ__init____mutmut_2': xǁLogisticModelǁ__init____mutmut_2, 
        'xǁLogisticModelǁ__init____mutmut_3': xǁLogisticModelǁ__init____mutmut_3, 
        'xǁLogisticModelǁ__init____mutmut_4': xǁLogisticModelǁ__init____mutmut_4, 
        'xǁLogisticModelǁ__init____mutmut_5': xǁLogisticModelǁ__init____mutmut_5
    }
    xǁLogisticModelǁ__init____mutmut_orig.__name__ = 'xǁLogisticModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        """Return the list of parameter names for the logistic model, including base parameters and covariate-specific coefficients.

        Returns
        -------
            names (Sequence[str]): List of parameter names, with covariate effects prefixed by 'beta_L_', 'beta_k_', and 'beta_x0_' for each covariate.
        """
        names = ["L", "k", "x0"]
        if self.t_event is not None:
            names.extend(["L_post", "k_post", "x0_post"])
        for cov in self.covariates:
            names.extend([f"beta_L_{cov}", f"beta_k_{cov}", f"beta_x0_{cov}"])
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = None
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "XXLXX": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "l": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) / 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(None) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 2.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "XXkXX": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "K": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 1.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "XXx0XX": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "X0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(None),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                None,
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "XXL_postXX": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "l_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_POST": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) / 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(None) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 2.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "XXk_postXX": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "K_POST": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 1.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "XXx0_postXX": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "X0_POST": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(None),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = None
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 1.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = None
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 1.0
            guesses[f"beta_x0_{cov}"] = 0.0
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = None
        return guesses

    def xǁLogisticModelǁinitial_guesses__mutmut_32(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {
            "L": np.max(y) * 1.1,
            "k": 0.1,
            "x0": np.median(t),
        }
        if self.t_event is not None:
            guesses.update(
                {
                    "L_post": np.max(y) * 1.1,
                    "k_post": 0.1,
                    "x0_post": np.median(t),
                },
            )
        for cov in self.covariates:
            guesses[f"beta_L_{cov}"] = 0.0
            guesses[f"beta_k_{cov}"] = 0.0
            guesses[f"beta_x0_{cov}"] = 1.0
        return guesses
    
    xǁLogisticModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁinitial_guesses__mutmut_1': xǁLogisticModelǁinitial_guesses__mutmut_1, 
        'xǁLogisticModelǁinitial_guesses__mutmut_2': xǁLogisticModelǁinitial_guesses__mutmut_2, 
        'xǁLogisticModelǁinitial_guesses__mutmut_3': xǁLogisticModelǁinitial_guesses__mutmut_3, 
        'xǁLogisticModelǁinitial_guesses__mutmut_4': xǁLogisticModelǁinitial_guesses__mutmut_4, 
        'xǁLogisticModelǁinitial_guesses__mutmut_5': xǁLogisticModelǁinitial_guesses__mutmut_5, 
        'xǁLogisticModelǁinitial_guesses__mutmut_6': xǁLogisticModelǁinitial_guesses__mutmut_6, 
        'xǁLogisticModelǁinitial_guesses__mutmut_7': xǁLogisticModelǁinitial_guesses__mutmut_7, 
        'xǁLogisticModelǁinitial_guesses__mutmut_8': xǁLogisticModelǁinitial_guesses__mutmut_8, 
        'xǁLogisticModelǁinitial_guesses__mutmut_9': xǁLogisticModelǁinitial_guesses__mutmut_9, 
        'xǁLogisticModelǁinitial_guesses__mutmut_10': xǁLogisticModelǁinitial_guesses__mutmut_10, 
        'xǁLogisticModelǁinitial_guesses__mutmut_11': xǁLogisticModelǁinitial_guesses__mutmut_11, 
        'xǁLogisticModelǁinitial_guesses__mutmut_12': xǁLogisticModelǁinitial_guesses__mutmut_12, 
        'xǁLogisticModelǁinitial_guesses__mutmut_13': xǁLogisticModelǁinitial_guesses__mutmut_13, 
        'xǁLogisticModelǁinitial_guesses__mutmut_14': xǁLogisticModelǁinitial_guesses__mutmut_14, 
        'xǁLogisticModelǁinitial_guesses__mutmut_15': xǁLogisticModelǁinitial_guesses__mutmut_15, 
        'xǁLogisticModelǁinitial_guesses__mutmut_16': xǁLogisticModelǁinitial_guesses__mutmut_16, 
        'xǁLogisticModelǁinitial_guesses__mutmut_17': xǁLogisticModelǁinitial_guesses__mutmut_17, 
        'xǁLogisticModelǁinitial_guesses__mutmut_18': xǁLogisticModelǁinitial_guesses__mutmut_18, 
        'xǁLogisticModelǁinitial_guesses__mutmut_19': xǁLogisticModelǁinitial_guesses__mutmut_19, 
        'xǁLogisticModelǁinitial_guesses__mutmut_20': xǁLogisticModelǁinitial_guesses__mutmut_20, 
        'xǁLogisticModelǁinitial_guesses__mutmut_21': xǁLogisticModelǁinitial_guesses__mutmut_21, 
        'xǁLogisticModelǁinitial_guesses__mutmut_22': xǁLogisticModelǁinitial_guesses__mutmut_22, 
        'xǁLogisticModelǁinitial_guesses__mutmut_23': xǁLogisticModelǁinitial_guesses__mutmut_23, 
        'xǁLogisticModelǁinitial_guesses__mutmut_24': xǁLogisticModelǁinitial_guesses__mutmut_24, 
        'xǁLogisticModelǁinitial_guesses__mutmut_25': xǁLogisticModelǁinitial_guesses__mutmut_25, 
        'xǁLogisticModelǁinitial_guesses__mutmut_26': xǁLogisticModelǁinitial_guesses__mutmut_26, 
        'xǁLogisticModelǁinitial_guesses__mutmut_27': xǁLogisticModelǁinitial_guesses__mutmut_27, 
        'xǁLogisticModelǁinitial_guesses__mutmut_28': xǁLogisticModelǁinitial_guesses__mutmut_28, 
        'xǁLogisticModelǁinitial_guesses__mutmut_29': xǁLogisticModelǁinitial_guesses__mutmut_29, 
        'xǁLogisticModelǁinitial_guesses__mutmut_30': xǁLogisticModelǁinitial_guesses__mutmut_30, 
        'xǁLogisticModelǁinitial_guesses__mutmut_31': xǁLogisticModelǁinitial_guesses__mutmut_31, 
        'xǁLogisticModelǁinitial_guesses__mutmut_32': xǁLogisticModelǁinitial_guesses__mutmut_32
    }
    xǁLogisticModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁLogisticModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = None
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "XXLXX": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "l": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(None), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "XXkXX": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "K": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1.000001, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "XXx0XX": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "X0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (+np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                None,
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "XXL_postXX": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "l_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_POST": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(None), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "XXk_postXX": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "K_POST": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1.000001, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "XXx0_postXX": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "X0_POST": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (+np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = None
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (+np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = None
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (+np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁLogisticModelǁbounds__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = None
        return bounds

    def xǁLogisticModelǁbounds__mutmut_28(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Return parameter bounds for the logistic model, including covariate effects.

        Parameters
        ----------
            t (Sequence[float]): Time points of the observations.
            y (Sequence[float]): Observed values corresponding to each time point.

        Returns
        -------
            Dict[str, tuple]: Dictionary mapping parameter names to their (lower, upper) bounds.
        """
        bounds = {
            "L": (np.max(y), np.inf),
            "k": (1e-6, np.inf),
            "x0": (-np.inf, np.inf),
        }
        if self.t_event is not None:
            bounds.update(
                {
                    "L_post": (np.max(y), np.inf),
                    "k_post": (1e-6, np.inf),
                    "x0_post": (-np.inf, np.inf),
                },
            )
        for cov in self.covariates:
            bounds[f"beta_L_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_k_{cov}"] = (-np.inf, np.inf)
            bounds[f"beta_x0_{cov}"] = (+np.inf, np.inf)
        return bounds
    
    xǁLogisticModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁbounds__mutmut_1': xǁLogisticModelǁbounds__mutmut_1, 
        'xǁLogisticModelǁbounds__mutmut_2': xǁLogisticModelǁbounds__mutmut_2, 
        'xǁLogisticModelǁbounds__mutmut_3': xǁLogisticModelǁbounds__mutmut_3, 
        'xǁLogisticModelǁbounds__mutmut_4': xǁLogisticModelǁbounds__mutmut_4, 
        'xǁLogisticModelǁbounds__mutmut_5': xǁLogisticModelǁbounds__mutmut_5, 
        'xǁLogisticModelǁbounds__mutmut_6': xǁLogisticModelǁbounds__mutmut_6, 
        'xǁLogisticModelǁbounds__mutmut_7': xǁLogisticModelǁbounds__mutmut_7, 
        'xǁLogisticModelǁbounds__mutmut_8': xǁLogisticModelǁbounds__mutmut_8, 
        'xǁLogisticModelǁbounds__mutmut_9': xǁLogisticModelǁbounds__mutmut_9, 
        'xǁLogisticModelǁbounds__mutmut_10': xǁLogisticModelǁbounds__mutmut_10, 
        'xǁLogisticModelǁbounds__mutmut_11': xǁLogisticModelǁbounds__mutmut_11, 
        'xǁLogisticModelǁbounds__mutmut_12': xǁLogisticModelǁbounds__mutmut_12, 
        'xǁLogisticModelǁbounds__mutmut_13': xǁLogisticModelǁbounds__mutmut_13, 
        'xǁLogisticModelǁbounds__mutmut_14': xǁLogisticModelǁbounds__mutmut_14, 
        'xǁLogisticModelǁbounds__mutmut_15': xǁLogisticModelǁbounds__mutmut_15, 
        'xǁLogisticModelǁbounds__mutmut_16': xǁLogisticModelǁbounds__mutmut_16, 
        'xǁLogisticModelǁbounds__mutmut_17': xǁLogisticModelǁbounds__mutmut_17, 
        'xǁLogisticModelǁbounds__mutmut_18': xǁLogisticModelǁbounds__mutmut_18, 
        'xǁLogisticModelǁbounds__mutmut_19': xǁLogisticModelǁbounds__mutmut_19, 
        'xǁLogisticModelǁbounds__mutmut_20': xǁLogisticModelǁbounds__mutmut_20, 
        'xǁLogisticModelǁbounds__mutmut_21': xǁLogisticModelǁbounds__mutmut_21, 
        'xǁLogisticModelǁbounds__mutmut_22': xǁLogisticModelǁbounds__mutmut_22, 
        'xǁLogisticModelǁbounds__mutmut_23': xǁLogisticModelǁbounds__mutmut_23, 
        'xǁLogisticModelǁbounds__mutmut_24': xǁLogisticModelǁbounds__mutmut_24, 
        'xǁLogisticModelǁbounds__mutmut_25': xǁLogisticModelǁbounds__mutmut_25, 
        'xǁLogisticModelǁbounds__mutmut_26': xǁLogisticModelǁbounds__mutmut_26, 
        'xǁLogisticModelǁbounds__mutmut_27': xǁLogisticModelǁbounds__mutmut_27, 
        'xǁLogisticModelǁbounds__mutmut_28': xǁLogisticModelǁbounds__mutmut_28
    }
    xǁLogisticModelǁbounds__mutmut_orig.__name__ = 'xǁLogisticModelǁbounds'

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁpredict__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError(None)

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = None

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(None)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = None
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr <= self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = None

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = None

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(None)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(None):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = None
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["XXLXX"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["l"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = None
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["XXkXX"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["K"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = None
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["XXx0XX"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["X0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = None

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L * (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 - backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (2 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(None))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k / (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(+k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_32(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] + x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_33(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(None):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_34(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = None
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_35(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["XXL_postXX"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_36(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["l_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_37(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_POST"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_38(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = None
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_39(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["XXk_postXX"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_40(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["K_POST"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_41(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = None
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_42(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["XXx0_postXX"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_43(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["X0_POST"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_44(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = None

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_45(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L * (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_46(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 - backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_47(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (2 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_48(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(None))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_49(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k / (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_50(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(+k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_51(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] + x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_52(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = None
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_53(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["XXLXX"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_54(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["l"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_55(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = None
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_56(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["XXkXX"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_57(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["K"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_58(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = None

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_59(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["XXx0XX"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_60(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["X0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_61(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = None

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_62(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(None, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_63(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, None, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_64(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, None)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_65(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_66(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_67(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, )

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_68(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L = self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_69(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L -= self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_70(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] / cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_71(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k = self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_72(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k -= self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_73(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] / cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_74(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 = self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_75(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 -= self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_76(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] / cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_77(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L * (1 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_78(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 - backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_79(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (2 + backend.current_backend.exp(-k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_80(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(None))

    def xǁLogisticModelǁpredict__mutmut_81(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k / (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_82(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(+k * (t_arr - x0)))

    def xǁLogisticModelǁpredict__mutmut_83(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Predicts the cumulative values of the logistic diffusion process at specified time points.

        Parameters
        ----------
            t (Sequence[float]): Time points at which to compute predictions.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            Sequence[float]: Predicted cumulative values of the logistic model at each time point.

        Raises
        ------
            RuntimeError: If the model parameters have not been set (i.e., the model is not fitted).
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = backend.current_backend.array(t)

        if self.t_event is not None:
            pre_event_mask = t_arr < self.t_event
            post_event_mask = ~pre_event_mask

            y_pred = backend.current_backend.zeros_like(t_arr)

            if backend.current_backend.any(pre_event_mask):
                L = self._params["L"]
                k = self._params["k"]
                x0 = self._params["x0"]
                y_pred[pre_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[pre_event_mask] - x0)))

            if backend.current_backend.any(post_event_mask):
                L = self._params["L_post"]
                k = self._params["k_post"]
                x0 = self._params["x0_post"]
                y_pred[post_event_mask] = L / (1 + backend.current_backend.exp(-k * (t_arr[post_event_mask] - x0)))

            return y_pred

        L = self._params["L"]
        k = self._params["k"]
        x0 = self._params["x0"]

        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)

                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t
                x0 += self._params[f"beta_x0_{cov_name}"] * cov_val_t

        return L / (1 + backend.current_backend.exp(-k * (t_arr + x0)))
    
    xǁLogisticModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁpredict__mutmut_1': xǁLogisticModelǁpredict__mutmut_1, 
        'xǁLogisticModelǁpredict__mutmut_2': xǁLogisticModelǁpredict__mutmut_2, 
        'xǁLogisticModelǁpredict__mutmut_3': xǁLogisticModelǁpredict__mutmut_3, 
        'xǁLogisticModelǁpredict__mutmut_4': xǁLogisticModelǁpredict__mutmut_4, 
        'xǁLogisticModelǁpredict__mutmut_5': xǁLogisticModelǁpredict__mutmut_5, 
        'xǁLogisticModelǁpredict__mutmut_6': xǁLogisticModelǁpredict__mutmut_6, 
        'xǁLogisticModelǁpredict__mutmut_7': xǁLogisticModelǁpredict__mutmut_7, 
        'xǁLogisticModelǁpredict__mutmut_8': xǁLogisticModelǁpredict__mutmut_8, 
        'xǁLogisticModelǁpredict__mutmut_9': xǁLogisticModelǁpredict__mutmut_9, 
        'xǁLogisticModelǁpredict__mutmut_10': xǁLogisticModelǁpredict__mutmut_10, 
        'xǁLogisticModelǁpredict__mutmut_11': xǁLogisticModelǁpredict__mutmut_11, 
        'xǁLogisticModelǁpredict__mutmut_12': xǁLogisticModelǁpredict__mutmut_12, 
        'xǁLogisticModelǁpredict__mutmut_13': xǁLogisticModelǁpredict__mutmut_13, 
        'xǁLogisticModelǁpredict__mutmut_14': xǁLogisticModelǁpredict__mutmut_14, 
        'xǁLogisticModelǁpredict__mutmut_15': xǁLogisticModelǁpredict__mutmut_15, 
        'xǁLogisticModelǁpredict__mutmut_16': xǁLogisticModelǁpredict__mutmut_16, 
        'xǁLogisticModelǁpredict__mutmut_17': xǁLogisticModelǁpredict__mutmut_17, 
        'xǁLogisticModelǁpredict__mutmut_18': xǁLogisticModelǁpredict__mutmut_18, 
        'xǁLogisticModelǁpredict__mutmut_19': xǁLogisticModelǁpredict__mutmut_19, 
        'xǁLogisticModelǁpredict__mutmut_20': xǁLogisticModelǁpredict__mutmut_20, 
        'xǁLogisticModelǁpredict__mutmut_21': xǁLogisticModelǁpredict__mutmut_21, 
        'xǁLogisticModelǁpredict__mutmut_22': xǁLogisticModelǁpredict__mutmut_22, 
        'xǁLogisticModelǁpredict__mutmut_23': xǁLogisticModelǁpredict__mutmut_23, 
        'xǁLogisticModelǁpredict__mutmut_24': xǁLogisticModelǁpredict__mutmut_24, 
        'xǁLogisticModelǁpredict__mutmut_25': xǁLogisticModelǁpredict__mutmut_25, 
        'xǁLogisticModelǁpredict__mutmut_26': xǁLogisticModelǁpredict__mutmut_26, 
        'xǁLogisticModelǁpredict__mutmut_27': xǁLogisticModelǁpredict__mutmut_27, 
        'xǁLogisticModelǁpredict__mutmut_28': xǁLogisticModelǁpredict__mutmut_28, 
        'xǁLogisticModelǁpredict__mutmut_29': xǁLogisticModelǁpredict__mutmut_29, 
        'xǁLogisticModelǁpredict__mutmut_30': xǁLogisticModelǁpredict__mutmut_30, 
        'xǁLogisticModelǁpredict__mutmut_31': xǁLogisticModelǁpredict__mutmut_31, 
        'xǁLogisticModelǁpredict__mutmut_32': xǁLogisticModelǁpredict__mutmut_32, 
        'xǁLogisticModelǁpredict__mutmut_33': xǁLogisticModelǁpredict__mutmut_33, 
        'xǁLogisticModelǁpredict__mutmut_34': xǁLogisticModelǁpredict__mutmut_34, 
        'xǁLogisticModelǁpredict__mutmut_35': xǁLogisticModelǁpredict__mutmut_35, 
        'xǁLogisticModelǁpredict__mutmut_36': xǁLogisticModelǁpredict__mutmut_36, 
        'xǁLogisticModelǁpredict__mutmut_37': xǁLogisticModelǁpredict__mutmut_37, 
        'xǁLogisticModelǁpredict__mutmut_38': xǁLogisticModelǁpredict__mutmut_38, 
        'xǁLogisticModelǁpredict__mutmut_39': xǁLogisticModelǁpredict__mutmut_39, 
        'xǁLogisticModelǁpredict__mutmut_40': xǁLogisticModelǁpredict__mutmut_40, 
        'xǁLogisticModelǁpredict__mutmut_41': xǁLogisticModelǁpredict__mutmut_41, 
        'xǁLogisticModelǁpredict__mutmut_42': xǁLogisticModelǁpredict__mutmut_42, 
        'xǁLogisticModelǁpredict__mutmut_43': xǁLogisticModelǁpredict__mutmut_43, 
        'xǁLogisticModelǁpredict__mutmut_44': xǁLogisticModelǁpredict__mutmut_44, 
        'xǁLogisticModelǁpredict__mutmut_45': xǁLogisticModelǁpredict__mutmut_45, 
        'xǁLogisticModelǁpredict__mutmut_46': xǁLogisticModelǁpredict__mutmut_46, 
        'xǁLogisticModelǁpredict__mutmut_47': xǁLogisticModelǁpredict__mutmut_47, 
        'xǁLogisticModelǁpredict__mutmut_48': xǁLogisticModelǁpredict__mutmut_48, 
        'xǁLogisticModelǁpredict__mutmut_49': xǁLogisticModelǁpredict__mutmut_49, 
        'xǁLogisticModelǁpredict__mutmut_50': xǁLogisticModelǁpredict__mutmut_50, 
        'xǁLogisticModelǁpredict__mutmut_51': xǁLogisticModelǁpredict__mutmut_51, 
        'xǁLogisticModelǁpredict__mutmut_52': xǁLogisticModelǁpredict__mutmut_52, 
        'xǁLogisticModelǁpredict__mutmut_53': xǁLogisticModelǁpredict__mutmut_53, 
        'xǁLogisticModelǁpredict__mutmut_54': xǁLogisticModelǁpredict__mutmut_54, 
        'xǁLogisticModelǁpredict__mutmut_55': xǁLogisticModelǁpredict__mutmut_55, 
        'xǁLogisticModelǁpredict__mutmut_56': xǁLogisticModelǁpredict__mutmut_56, 
        'xǁLogisticModelǁpredict__mutmut_57': xǁLogisticModelǁpredict__mutmut_57, 
        'xǁLogisticModelǁpredict__mutmut_58': xǁLogisticModelǁpredict__mutmut_58, 
        'xǁLogisticModelǁpredict__mutmut_59': xǁLogisticModelǁpredict__mutmut_59, 
        'xǁLogisticModelǁpredict__mutmut_60': xǁLogisticModelǁpredict__mutmut_60, 
        'xǁLogisticModelǁpredict__mutmut_61': xǁLogisticModelǁpredict__mutmut_61, 
        'xǁLogisticModelǁpredict__mutmut_62': xǁLogisticModelǁpredict__mutmut_62, 
        'xǁLogisticModelǁpredict__mutmut_63': xǁLogisticModelǁpredict__mutmut_63, 
        'xǁLogisticModelǁpredict__mutmut_64': xǁLogisticModelǁpredict__mutmut_64, 
        'xǁLogisticModelǁpredict__mutmut_65': xǁLogisticModelǁpredict__mutmut_65, 
        'xǁLogisticModelǁpredict__mutmut_66': xǁLogisticModelǁpredict__mutmut_66, 
        'xǁLogisticModelǁpredict__mutmut_67': xǁLogisticModelǁpredict__mutmut_67, 
        'xǁLogisticModelǁpredict__mutmut_68': xǁLogisticModelǁpredict__mutmut_68, 
        'xǁLogisticModelǁpredict__mutmut_69': xǁLogisticModelǁpredict__mutmut_69, 
        'xǁLogisticModelǁpredict__mutmut_70': xǁLogisticModelǁpredict__mutmut_70, 
        'xǁLogisticModelǁpredict__mutmut_71': xǁLogisticModelǁpredict__mutmut_71, 
        'xǁLogisticModelǁpredict__mutmut_72': xǁLogisticModelǁpredict__mutmut_72, 
        'xǁLogisticModelǁpredict__mutmut_73': xǁLogisticModelǁpredict__mutmut_73, 
        'xǁLogisticModelǁpredict__mutmut_74': xǁLogisticModelǁpredict__mutmut_74, 
        'xǁLogisticModelǁpredict__mutmut_75': xǁLogisticModelǁpredict__mutmut_75, 
        'xǁLogisticModelǁpredict__mutmut_76': xǁLogisticModelǁpredict__mutmut_76, 
        'xǁLogisticModelǁpredict__mutmut_77': xǁLogisticModelǁpredict__mutmut_77, 
        'xǁLogisticModelǁpredict__mutmut_78': xǁLogisticModelǁpredict__mutmut_78, 
        'xǁLogisticModelǁpredict__mutmut_79': xǁLogisticModelǁpredict__mutmut_79, 
        'xǁLogisticModelǁpredict__mutmut_80': xǁLogisticModelǁpredict__mutmut_80, 
        'xǁLogisticModelǁpredict__mutmut_81': xǁLogisticModelǁpredict__mutmut_81, 
        'xǁLogisticModelǁpredict__mutmut_82': xǁLogisticModelǁpredict__mutmut_82, 
        'xǁLogisticModelǁpredict__mutmut_83': xǁLogisticModelǁpredict__mutmut_83
    }
    xǁLogisticModelǁpredict__mutmut_orig.__name__ = 'xǁLogisticModelǁpredict'

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        args = [t, y, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁscore__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = None
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(None, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, None)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, )
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = None
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            None,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) * 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) + y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(None) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 3,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            None,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) * 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) + backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(None) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(None)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 3,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁLogisticModelǁscore__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁLogisticModelǁscore__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Compute the coefficient of determination (R²) between observed values and model predictions.

        Parameters
        ----------
            t (Sequence[float]): Time points at which observations were made.
            y (Sequence[float]): Observed values corresponding to time points.
            covariates (Dict[str, Sequence[float]], optional): Covariate values for each time point.

        Returns
        -------
            float: The R² score indicating the proportion of variance explained by the model predictions.

        Raises
        ------
            RuntimeError: If the model has not been fitted.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = backend.current_backend.sum(
            (backend.current_backend.array(y) - y_pred) ** 2,
        )
        ss_tot = backend.current_backend.sum(
            (backend.current_backend.array(y) - backend.current_backend.mean(y)) ** 2,
        )
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁLogisticModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁscore__mutmut_1': xǁLogisticModelǁscore__mutmut_1, 
        'xǁLogisticModelǁscore__mutmut_2': xǁLogisticModelǁscore__mutmut_2, 
        'xǁLogisticModelǁscore__mutmut_3': xǁLogisticModelǁscore__mutmut_3, 
        'xǁLogisticModelǁscore__mutmut_4': xǁLogisticModelǁscore__mutmut_4, 
        'xǁLogisticModelǁscore__mutmut_5': xǁLogisticModelǁscore__mutmut_5, 
        'xǁLogisticModelǁscore__mutmut_6': xǁLogisticModelǁscore__mutmut_6, 
        'xǁLogisticModelǁscore__mutmut_7': xǁLogisticModelǁscore__mutmut_7, 
        'xǁLogisticModelǁscore__mutmut_8': xǁLogisticModelǁscore__mutmut_8, 
        'xǁLogisticModelǁscore__mutmut_9': xǁLogisticModelǁscore__mutmut_9, 
        'xǁLogisticModelǁscore__mutmut_10': xǁLogisticModelǁscore__mutmut_10, 
        'xǁLogisticModelǁscore__mutmut_11': xǁLogisticModelǁscore__mutmut_11, 
        'xǁLogisticModelǁscore__mutmut_12': xǁLogisticModelǁscore__mutmut_12, 
        'xǁLogisticModelǁscore__mutmut_13': xǁLogisticModelǁscore__mutmut_13, 
        'xǁLogisticModelǁscore__mutmut_14': xǁLogisticModelǁscore__mutmut_14, 
        'xǁLogisticModelǁscore__mutmut_15': xǁLogisticModelǁscore__mutmut_15, 
        'xǁLogisticModelǁscore__mutmut_16': xǁLogisticModelǁscore__mutmut_16, 
        'xǁLogisticModelǁscore__mutmut_17': xǁLogisticModelǁscore__mutmut_17, 
        'xǁLogisticModelǁscore__mutmut_18': xǁLogisticModelǁscore__mutmut_18, 
        'xǁLogisticModelǁscore__mutmut_19': xǁLogisticModelǁscore__mutmut_19, 
        'xǁLogisticModelǁscore__mutmut_20': xǁLogisticModelǁscore__mutmut_20, 
        'xǁLogisticModelǁscore__mutmut_21': xǁLogisticModelǁscore__mutmut_21, 
        'xǁLogisticModelǁscore__mutmut_22': xǁLogisticModelǁscore__mutmut_22, 
        'xǁLogisticModelǁscore__mutmut_23': xǁLogisticModelǁscore__mutmut_23, 
        'xǁLogisticModelǁscore__mutmut_24': xǁLogisticModelǁscore__mutmut_24, 
        'xǁLogisticModelǁscore__mutmut_25': xǁLogisticModelǁscore__mutmut_25, 
        'xǁLogisticModelǁscore__mutmut_26': xǁLogisticModelǁscore__mutmut_26, 
        'xǁLogisticModelǁscore__mutmut_27': xǁLogisticModelǁscore__mutmut_27, 
        'xǁLogisticModelǁscore__mutmut_28': xǁLogisticModelǁscore__mutmut_28, 
        'xǁLogisticModelǁscore__mutmut_29': xǁLogisticModelǁscore__mutmut_29
    }
    xǁLogisticModelǁscore__mutmut_orig.__name__ = 'xǁLogisticModelǁscore'

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
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError(None)

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = None

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(None, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, None)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, )

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = None
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["XXLXX"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["l"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = None
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["XXkXX"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["K"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = None
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(None, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, None, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, None)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, )
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L = self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L -= self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] / cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k = self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k -= self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] / cov_val_t

        return k * y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred / (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k / y_pred * (1 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_32(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 + y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_33(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (2 - y_pred / L)

    def xǁLogisticModelǁpredict_adoption_rate__mutmut_34(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)

        # The adoption rate is the derivative of the cumulative adoption
        # For the logistic function, the derivative is: k * y * (1 - y/L)
        L = self._params["L"]
        k = self._params["k"]
        if covariates:
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t, cov_values)
                L += self._params[f"beta_L_{cov_name}"] * cov_val_t
                k += self._params[f"beta_k_{cov_name}"] * cov_val_t

        return k * y_pred * (1 - y_pred * L)
    
    xǁLogisticModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁpredict_adoption_rate__mutmut_1': xǁLogisticModelǁpredict_adoption_rate__mutmut_1, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_2': xǁLogisticModelǁpredict_adoption_rate__mutmut_2, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_3': xǁLogisticModelǁpredict_adoption_rate__mutmut_3, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_4': xǁLogisticModelǁpredict_adoption_rate__mutmut_4, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_5': xǁLogisticModelǁpredict_adoption_rate__mutmut_5, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_6': xǁLogisticModelǁpredict_adoption_rate__mutmut_6, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_7': xǁLogisticModelǁpredict_adoption_rate__mutmut_7, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_8': xǁLogisticModelǁpredict_adoption_rate__mutmut_8, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_9': xǁLogisticModelǁpredict_adoption_rate__mutmut_9, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_10': xǁLogisticModelǁpredict_adoption_rate__mutmut_10, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_11': xǁLogisticModelǁpredict_adoption_rate__mutmut_11, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_12': xǁLogisticModelǁpredict_adoption_rate__mutmut_12, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_13': xǁLogisticModelǁpredict_adoption_rate__mutmut_13, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_14': xǁLogisticModelǁpredict_adoption_rate__mutmut_14, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_15': xǁLogisticModelǁpredict_adoption_rate__mutmut_15, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_16': xǁLogisticModelǁpredict_adoption_rate__mutmut_16, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_17': xǁLogisticModelǁpredict_adoption_rate__mutmut_17, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_18': xǁLogisticModelǁpredict_adoption_rate__mutmut_18, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_19': xǁLogisticModelǁpredict_adoption_rate__mutmut_19, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_20': xǁLogisticModelǁpredict_adoption_rate__mutmut_20, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_21': xǁLogisticModelǁpredict_adoption_rate__mutmut_21, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_22': xǁLogisticModelǁpredict_adoption_rate__mutmut_22, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_23': xǁLogisticModelǁpredict_adoption_rate__mutmut_23, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_24': xǁLogisticModelǁpredict_adoption_rate__mutmut_24, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_25': xǁLogisticModelǁpredict_adoption_rate__mutmut_25, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_26': xǁLogisticModelǁpredict_adoption_rate__mutmut_26, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_27': xǁLogisticModelǁpredict_adoption_rate__mutmut_27, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_28': xǁLogisticModelǁpredict_adoption_rate__mutmut_28, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_29': xǁLogisticModelǁpredict_adoption_rate__mutmut_29, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_30': xǁLogisticModelǁpredict_adoption_rate__mutmut_30, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_31': xǁLogisticModelǁpredict_adoption_rate__mutmut_31, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_32': xǁLogisticModelǁpredict_adoption_rate__mutmut_32, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_33': xǁLogisticModelǁpredict_adoption_rate__mutmut_33, 
        'xǁLogisticModelǁpredict_adoption_rate__mutmut_34': xǁLogisticModelǁpredict_adoption_rate__mutmut_34
    }
    xǁLogisticModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁLogisticModelǁpredict_adoption_rate'

    def cumulative_adoption(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        args = [t, *params]# type: ignore
        kwargs = {**param_kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁcumulative_adoption__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁcumulative_adoption__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁcumulative_adoption__mutmut_orig(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_1(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = None
        else:
            self.params_ = dict(zip(self.param_names, params))
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_2(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = None
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_3(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(None)
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_4(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(None, params))
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_5(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(self.param_names, None))
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_6(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(params))
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_7(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(self.param_names, ))
        return self.predict(t)

    def xǁLogisticModelǁcumulative_adoption__mutmut_8(
        self,
        t: Sequence[float],
        *params,
        **param_kwargs,
    ) -> Sequence[float]:
        if param_kwargs:
            self.params_ = param_kwargs
        else:
            self.params_ = dict(zip(self.param_names, params))
        return self.predict(None)
    
    xǁLogisticModelǁcumulative_adoption__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁcumulative_adoption__mutmut_1': xǁLogisticModelǁcumulative_adoption__mutmut_1, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_2': xǁLogisticModelǁcumulative_adoption__mutmut_2, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_3': xǁLogisticModelǁcumulative_adoption__mutmut_3, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_4': xǁLogisticModelǁcumulative_adoption__mutmut_4, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_5': xǁLogisticModelǁcumulative_adoption__mutmut_5, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_6': xǁLogisticModelǁcumulative_adoption__mutmut_6, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_7': xǁLogisticModelǁcumulative_adoption__mutmut_7, 
        'xǁLogisticModelǁcumulative_adoption__mutmut_8': xǁLogisticModelǁcumulative_adoption__mutmut_8
    }
    xǁLogisticModelǁcumulative_adoption__mutmut_orig.__name__ = 'xǁLogisticModelǁcumulative_adoption'

    def differential_equation(self, t, y, params, covariates, t_eval):
        args = [t, y, params, covariates, t_eval]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁLogisticModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁLogisticModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁLogisticModelǁdifferential_equation__mutmut_orig(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_1(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None or t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_2(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_3(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t > self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_4(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = None
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_5(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[4], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_6(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[5], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_7(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[6]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_8(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = None
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_9(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 4
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_10(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = None
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_11(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[1], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_12(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[2], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_13(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[3]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_14(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = None

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_15(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 1

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_16(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = None
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_17(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 - param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_18(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 4 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_19(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = None
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_20(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(None, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_21(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, None, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_22(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, None)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_23(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_24(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_25(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, )
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_26(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L = params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_27(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L -= params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_28(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] / cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_29(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k = params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_30(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k -= params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_31(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] / cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_32(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx - 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_33(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 2] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_34(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 = params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_35(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 -= params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_36(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] / cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_37(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx - 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_38(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 3] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_39(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx = 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_40(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx -= 3

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_41(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 4

        return k * y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_42(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] / (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_43(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k / y[0] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_44(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[1] * (1 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_45(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 + y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_46(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (2 - y[0] / L)

    def xǁLogisticModelǁdifferential_equation__mutmut_47(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[0] * L)

    def xǁLogisticModelǁdifferential_equation__mutmut_48(self, t, y, params, covariates, t_eval):
        """Differential equation for the logistic model."""
        if self.t_event is not None and t >= self.t_event:
            L, k, x0 = params[3], params[4], params[5]
            param_idx_offset = 3
        else:
            L, k, x0 = params[0], params[1], params[2]
            param_idx_offset = 0

        if covariates:
            param_idx = 3 + param_idx_offset
            for cov_name, cov_values in covariates.items():
                cov_val_t = backend.current_backend.interp(t, t_eval, cov_values)
                L += params[param_idx] * cov_val_t
                k += params[param_idx + 1] * cov_val_t
                x0 += params[param_idx + 2] * cov_val_t
                param_idx += 3

        return k * y[0] * (1 - y[1] / L)
    
    xǁLogisticModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁLogisticModelǁdifferential_equation__mutmut_1': xǁLogisticModelǁdifferential_equation__mutmut_1, 
        'xǁLogisticModelǁdifferential_equation__mutmut_2': xǁLogisticModelǁdifferential_equation__mutmut_2, 
        'xǁLogisticModelǁdifferential_equation__mutmut_3': xǁLogisticModelǁdifferential_equation__mutmut_3, 
        'xǁLogisticModelǁdifferential_equation__mutmut_4': xǁLogisticModelǁdifferential_equation__mutmut_4, 
        'xǁLogisticModelǁdifferential_equation__mutmut_5': xǁLogisticModelǁdifferential_equation__mutmut_5, 
        'xǁLogisticModelǁdifferential_equation__mutmut_6': xǁLogisticModelǁdifferential_equation__mutmut_6, 
        'xǁLogisticModelǁdifferential_equation__mutmut_7': xǁLogisticModelǁdifferential_equation__mutmut_7, 
        'xǁLogisticModelǁdifferential_equation__mutmut_8': xǁLogisticModelǁdifferential_equation__mutmut_8, 
        'xǁLogisticModelǁdifferential_equation__mutmut_9': xǁLogisticModelǁdifferential_equation__mutmut_9, 
        'xǁLogisticModelǁdifferential_equation__mutmut_10': xǁLogisticModelǁdifferential_equation__mutmut_10, 
        'xǁLogisticModelǁdifferential_equation__mutmut_11': xǁLogisticModelǁdifferential_equation__mutmut_11, 
        'xǁLogisticModelǁdifferential_equation__mutmut_12': xǁLogisticModelǁdifferential_equation__mutmut_12, 
        'xǁLogisticModelǁdifferential_equation__mutmut_13': xǁLogisticModelǁdifferential_equation__mutmut_13, 
        'xǁLogisticModelǁdifferential_equation__mutmut_14': xǁLogisticModelǁdifferential_equation__mutmut_14, 
        'xǁLogisticModelǁdifferential_equation__mutmut_15': xǁLogisticModelǁdifferential_equation__mutmut_15, 
        'xǁLogisticModelǁdifferential_equation__mutmut_16': xǁLogisticModelǁdifferential_equation__mutmut_16, 
        'xǁLogisticModelǁdifferential_equation__mutmut_17': xǁLogisticModelǁdifferential_equation__mutmut_17, 
        'xǁLogisticModelǁdifferential_equation__mutmut_18': xǁLogisticModelǁdifferential_equation__mutmut_18, 
        'xǁLogisticModelǁdifferential_equation__mutmut_19': xǁLogisticModelǁdifferential_equation__mutmut_19, 
        'xǁLogisticModelǁdifferential_equation__mutmut_20': xǁLogisticModelǁdifferential_equation__mutmut_20, 
        'xǁLogisticModelǁdifferential_equation__mutmut_21': xǁLogisticModelǁdifferential_equation__mutmut_21, 
        'xǁLogisticModelǁdifferential_equation__mutmut_22': xǁLogisticModelǁdifferential_equation__mutmut_22, 
        'xǁLogisticModelǁdifferential_equation__mutmut_23': xǁLogisticModelǁdifferential_equation__mutmut_23, 
        'xǁLogisticModelǁdifferential_equation__mutmut_24': xǁLogisticModelǁdifferential_equation__mutmut_24, 
        'xǁLogisticModelǁdifferential_equation__mutmut_25': xǁLogisticModelǁdifferential_equation__mutmut_25, 
        'xǁLogisticModelǁdifferential_equation__mutmut_26': xǁLogisticModelǁdifferential_equation__mutmut_26, 
        'xǁLogisticModelǁdifferential_equation__mutmut_27': xǁLogisticModelǁdifferential_equation__mutmut_27, 
        'xǁLogisticModelǁdifferential_equation__mutmut_28': xǁLogisticModelǁdifferential_equation__mutmut_28, 
        'xǁLogisticModelǁdifferential_equation__mutmut_29': xǁLogisticModelǁdifferential_equation__mutmut_29, 
        'xǁLogisticModelǁdifferential_equation__mutmut_30': xǁLogisticModelǁdifferential_equation__mutmut_30, 
        'xǁLogisticModelǁdifferential_equation__mutmut_31': xǁLogisticModelǁdifferential_equation__mutmut_31, 
        'xǁLogisticModelǁdifferential_equation__mutmut_32': xǁLogisticModelǁdifferential_equation__mutmut_32, 
        'xǁLogisticModelǁdifferential_equation__mutmut_33': xǁLogisticModelǁdifferential_equation__mutmut_33, 
        'xǁLogisticModelǁdifferential_equation__mutmut_34': xǁLogisticModelǁdifferential_equation__mutmut_34, 
        'xǁLogisticModelǁdifferential_equation__mutmut_35': xǁLogisticModelǁdifferential_equation__mutmut_35, 
        'xǁLogisticModelǁdifferential_equation__mutmut_36': xǁLogisticModelǁdifferential_equation__mutmut_36, 
        'xǁLogisticModelǁdifferential_equation__mutmut_37': xǁLogisticModelǁdifferential_equation__mutmut_37, 
        'xǁLogisticModelǁdifferential_equation__mutmut_38': xǁLogisticModelǁdifferential_equation__mutmut_38, 
        'xǁLogisticModelǁdifferential_equation__mutmut_39': xǁLogisticModelǁdifferential_equation__mutmut_39, 
        'xǁLogisticModelǁdifferential_equation__mutmut_40': xǁLogisticModelǁdifferential_equation__mutmut_40, 
        'xǁLogisticModelǁdifferential_equation__mutmut_41': xǁLogisticModelǁdifferential_equation__mutmut_41, 
        'xǁLogisticModelǁdifferential_equation__mutmut_42': xǁLogisticModelǁdifferential_equation__mutmut_42, 
        'xǁLogisticModelǁdifferential_equation__mutmut_43': xǁLogisticModelǁdifferential_equation__mutmut_43, 
        'xǁLogisticModelǁdifferential_equation__mutmut_44': xǁLogisticModelǁdifferential_equation__mutmut_44, 
        'xǁLogisticModelǁdifferential_equation__mutmut_45': xǁLogisticModelǁdifferential_equation__mutmut_45, 
        'xǁLogisticModelǁdifferential_equation__mutmut_46': xǁLogisticModelǁdifferential_equation__mutmut_46, 
        'xǁLogisticModelǁdifferential_equation__mutmut_47': xǁLogisticModelǁdifferential_equation__mutmut_47, 
        'xǁLogisticModelǁdifferential_equation__mutmut_48': xǁLogisticModelǁdifferential_equation__mutmut_48
    }
    xǁLogisticModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁLogisticModelǁdifferential_equation'
