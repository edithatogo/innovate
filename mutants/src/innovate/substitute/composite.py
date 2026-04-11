# src/innovate/substitute/composite.py

from collections.abc import Sequence

import numpy as np

from innovate.backend import current_backend as B
from innovate.base.base import DiffusionModel
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


class CompositeDiffusionModel(DiffusionModel):
    """A generic model for the diffusion of multiple, potentially interacting products.
    This model is composed of multiple individual diffusion models and an interaction matrix
    that defines how the adoption of one product affects the adoption of others.
    """

    def __init__(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        args = [models, alpha]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁ__init____mutmut_orig(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_1(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = None
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_2(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = None
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_3(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = None

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_4(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is not None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_5(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = None
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_6(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros(None)
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_7(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape == (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_8(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    None,
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_9(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "XXInteraction matrix alpha must have shape (n_models, n_models).XX",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_10(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_11(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "INTERACTION MATRIX ALPHA MUST HAVE SHAPE (N_MODELS, N_MODELS).",
                )
            self.alpha = alpha

    def xǁCompositeDiffusionModelǁ__init____mutmut_12(
        self,
        models: list[DiffusionModel],
        alpha: np.ndarray | None = None,
    ):
        """Initializes the CompositeDiffusionModel.

        Parameters
        ----------
            models: A list of individual diffusion models.
            alpha: An interaction matrix where alpha[i, j] represents the effect of model j on model i.
        """
        self.models = models
        self.n_models = len(models)
        self._params: dict[str, float] = {}

        if alpha is None:
            # Default to no interaction
            self.alpha = np.zeros((self.n_models, self.n_models))
        else:
            if alpha.shape != (self.n_models, self.n_models):
                raise ValueError(
                    "Interaction matrix alpha must have shape (n_models, n_models).",
                )
            self.alpha = None
    
    xǁCompositeDiffusionModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁ__init____mutmut_1': xǁCompositeDiffusionModelǁ__init____mutmut_1, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_2': xǁCompositeDiffusionModelǁ__init____mutmut_2, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_3': xǁCompositeDiffusionModelǁ__init____mutmut_3, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_4': xǁCompositeDiffusionModelǁ__init____mutmut_4, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_5': xǁCompositeDiffusionModelǁ__init____mutmut_5, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_6': xǁCompositeDiffusionModelǁ__init____mutmut_6, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_7': xǁCompositeDiffusionModelǁ__init____mutmut_7, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_8': xǁCompositeDiffusionModelǁ__init____mutmut_8, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_9': xǁCompositeDiffusionModelǁ__init____mutmut_9, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_10': xǁCompositeDiffusionModelǁ__init____mutmut_10, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_11': xǁCompositeDiffusionModelǁ__init____mutmut_11, 
        'xǁCompositeDiffusionModelǁ__init____mutmut_12': xǁCompositeDiffusionModelǁ__init____mutmut_12
    }
    xǁCompositeDiffusionModelǁ__init____mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for i, model in enumerate(self.models):
            for param_name in model.param_names:
                names.append(f"{param_name}_{i + 1}")

        # Add interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    names.append(f"alpha_{i + 1}_{j + 1}")
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = None
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(None):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = None
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) >= 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 2 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = None
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(None, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, None)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, )
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "XXmXX" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "M" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" not in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = None
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["XXmXX"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["M"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) / 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(None) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 2.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "XXLXX" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "l" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" not in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = None

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["XXLXX"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["l"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) / 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(None) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 2.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = None

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i - 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 2}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_32(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(None):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_33(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(None):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_34(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i == j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_35(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = None
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_36(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i - 1}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_37(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 2}_{j + 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_38(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j - 1}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_39(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 2}"] = 0.0
        return guesses

    def xǁCompositeDiffusionModelǁinitial_guesses__mutmut_40(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            # Use the i-th column of y for the i-th model
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_guesses = model.initial_guesses(t, y_model)
            # Override market potential guess
            if "m" in model_guesses:
                model_guesses["m"] = np.max(y_model) * 1.1
            if "L" in model_guesses:
                model_guesses["L"] = np.max(y_model) * 1.1

            for param_name, value in model_guesses.items():
                guesses[f"{param_name}_{i + 1}"] = value

        # Initial guesses for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    guesses[f"alpha_{i + 1}_{j + 1}"] = 1.0
        return guesses
    
    xǁCompositeDiffusionModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_1': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_1, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_2': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_2, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_3': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_3, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_4': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_4, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_5': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_5, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_6': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_6, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_7': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_7, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_8': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_8, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_9': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_9, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_10': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_10, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_11': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_11, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_12': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_12, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_13': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_13, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_14': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_14, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_15': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_15, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_16': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_16, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_17': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_17, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_18': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_18, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_19': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_19, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_20': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_20, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_21': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_21, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_22': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_22, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_23': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_23, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_24': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_24, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_25': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_25, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_26': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_26, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_27': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_27, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_28': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_28, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_29': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_29, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_30': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_30, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_31': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_31, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_32': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_32, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_33': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_33, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_34': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_34, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_35': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_35, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_36': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_36, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_37': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_37, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_38': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_38, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_39': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_39, 
        'xǁCompositeDiffusionModelǁinitial_guesses__mutmut_40': xǁCompositeDiffusionModelǁinitial_guesses__mutmut_40
    }
    xǁCompositeDiffusionModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = None
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(None):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = None
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) >= 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 2 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = None
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(None, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, None)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, )
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "XXmXX" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "M" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" not in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = None
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["XXmXX"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["M"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(None), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "XXLXX" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "l" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" not in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = None

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["XXLXX"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["l"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(None), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = None

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i - 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 2}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_28(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(None):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_29(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(None):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_30(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i == j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_31(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = None
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_32(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i - 1}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_33(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 2}_{j + 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_34(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j - 1}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_35(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 2}"] = (-np.inf, np.inf)
        return bounds

    def xǁCompositeDiffusionModelǁbounds__mutmut_36(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            y_model = y[:, i] if len(y.shape) > 1 else y
            model_bounds = model.bounds(t, y_model)
            # Override market potential bounds
            if "m" in model_bounds:
                model_bounds["m"] = (np.max(y_model), np.inf)
            if "L" in model_bounds:
                model_bounds["L"] = (np.max(y_model), np.inf)

            for param_name, value in model_bounds.items():
                bounds[f"{param_name}_{i + 1}"] = value

        # Bounds for interaction parameters
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    bounds[f"alpha_{i + 1}_{j + 1}"] = (+np.inf, np.inf)
        return bounds
    
    xǁCompositeDiffusionModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁbounds__mutmut_1': xǁCompositeDiffusionModelǁbounds__mutmut_1, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_2': xǁCompositeDiffusionModelǁbounds__mutmut_2, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_3': xǁCompositeDiffusionModelǁbounds__mutmut_3, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_4': xǁCompositeDiffusionModelǁbounds__mutmut_4, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_5': xǁCompositeDiffusionModelǁbounds__mutmut_5, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_6': xǁCompositeDiffusionModelǁbounds__mutmut_6, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_7': xǁCompositeDiffusionModelǁbounds__mutmut_7, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_8': xǁCompositeDiffusionModelǁbounds__mutmut_8, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_9': xǁCompositeDiffusionModelǁbounds__mutmut_9, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_10': xǁCompositeDiffusionModelǁbounds__mutmut_10, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_11': xǁCompositeDiffusionModelǁbounds__mutmut_11, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_12': xǁCompositeDiffusionModelǁbounds__mutmut_12, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_13': xǁCompositeDiffusionModelǁbounds__mutmut_13, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_14': xǁCompositeDiffusionModelǁbounds__mutmut_14, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_15': xǁCompositeDiffusionModelǁbounds__mutmut_15, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_16': xǁCompositeDiffusionModelǁbounds__mutmut_16, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_17': xǁCompositeDiffusionModelǁbounds__mutmut_17, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_18': xǁCompositeDiffusionModelǁbounds__mutmut_18, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_19': xǁCompositeDiffusionModelǁbounds__mutmut_19, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_20': xǁCompositeDiffusionModelǁbounds__mutmut_20, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_21': xǁCompositeDiffusionModelǁbounds__mutmut_21, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_22': xǁCompositeDiffusionModelǁbounds__mutmut_22, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_23': xǁCompositeDiffusionModelǁbounds__mutmut_23, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_24': xǁCompositeDiffusionModelǁbounds__mutmut_24, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_25': xǁCompositeDiffusionModelǁbounds__mutmut_25, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_26': xǁCompositeDiffusionModelǁbounds__mutmut_26, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_27': xǁCompositeDiffusionModelǁbounds__mutmut_27, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_28': xǁCompositeDiffusionModelǁbounds__mutmut_28, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_29': xǁCompositeDiffusionModelǁbounds__mutmut_29, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_30': xǁCompositeDiffusionModelǁbounds__mutmut_30, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_31': xǁCompositeDiffusionModelǁbounds__mutmut_31, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_32': xǁCompositeDiffusionModelǁbounds__mutmut_32, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_33': xǁCompositeDiffusionModelǁbounds__mutmut_33, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_34': xǁCompositeDiffusionModelǁbounds__mutmut_34, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_35': xǁCompositeDiffusionModelǁbounds__mutmut_35, 
        'xǁCompositeDiffusionModelǁbounds__mutmut_36': xǁCompositeDiffusionModelǁbounds__mutmut_36
    }
    xǁCompositeDiffusionModelǁbounds__mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁbounds'

    def predict(self, t: Sequence[float]) -> Sequence[float]:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁpredict__mutmut_orig(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_1(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_2(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError(None)

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_3(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_4(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_5(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_6(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = None
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_7(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(None)
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_8(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(None, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_9(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, None, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_10(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, None)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_11(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_12(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_13(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, )

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_14(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = None

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_15(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = None
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_16(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            None,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_17(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            None,
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_18(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            None,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_19(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=None,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_20(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method=None,
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_21(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=None,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_22(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=None,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_23(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=None,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_24(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_25(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_26(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_27(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_28(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_29(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_30(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_31(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_32(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[1], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_33(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[+1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_34(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-2]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_35(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="XXBDFXX",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_36(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="bdf",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_37(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=False,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_38(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1.000001,
            atol=1e-6,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_39(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1.000001,
        )
        return sol.sol(t).T

    def xǁCompositeDiffusionModelǁpredict__mutmut_40(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the cumulative adoption for each product."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = np.zeros(len(self.models))
        from scipy.integrate import solve_ivp

        # Compile the differential equation if the parameters are pytensor variables
        # if any(isinstance(p, pt.TensorVariable) for p in self._params.values()):
        #     t_sym = pt.scalar("t")
        #     y_sym = pt.vector("y")
        #     params_sym = [pt.scalar(name) for name in self.param_names]

        #     dydt = self.differential_equation(t_sym, y_sym, params_sym)

        #     def fun_with_params(t, y):
        #         return fun(t, y, *param_values)

        #     fun = fun_with_params
        # else:

        def ode_func(t, y):
            return self.differential_equation(t, y, self._params)

        fun = ode_func

        sol = solve_ivp(
            fun,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
        )
        return sol.sol(None).T
    
    xǁCompositeDiffusionModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁpredict__mutmut_1': xǁCompositeDiffusionModelǁpredict__mutmut_1, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_2': xǁCompositeDiffusionModelǁpredict__mutmut_2, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_3': xǁCompositeDiffusionModelǁpredict__mutmut_3, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_4': xǁCompositeDiffusionModelǁpredict__mutmut_4, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_5': xǁCompositeDiffusionModelǁpredict__mutmut_5, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_6': xǁCompositeDiffusionModelǁpredict__mutmut_6, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_7': xǁCompositeDiffusionModelǁpredict__mutmut_7, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_8': xǁCompositeDiffusionModelǁpredict__mutmut_8, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_9': xǁCompositeDiffusionModelǁpredict__mutmut_9, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_10': xǁCompositeDiffusionModelǁpredict__mutmut_10, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_11': xǁCompositeDiffusionModelǁpredict__mutmut_11, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_12': xǁCompositeDiffusionModelǁpredict__mutmut_12, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_13': xǁCompositeDiffusionModelǁpredict__mutmut_13, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_14': xǁCompositeDiffusionModelǁpredict__mutmut_14, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_15': xǁCompositeDiffusionModelǁpredict__mutmut_15, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_16': xǁCompositeDiffusionModelǁpredict__mutmut_16, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_17': xǁCompositeDiffusionModelǁpredict__mutmut_17, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_18': xǁCompositeDiffusionModelǁpredict__mutmut_18, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_19': xǁCompositeDiffusionModelǁpredict__mutmut_19, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_20': xǁCompositeDiffusionModelǁpredict__mutmut_20, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_21': xǁCompositeDiffusionModelǁpredict__mutmut_21, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_22': xǁCompositeDiffusionModelǁpredict__mutmut_22, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_23': xǁCompositeDiffusionModelǁpredict__mutmut_23, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_24': xǁCompositeDiffusionModelǁpredict__mutmut_24, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_25': xǁCompositeDiffusionModelǁpredict__mutmut_25, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_26': xǁCompositeDiffusionModelǁpredict__mutmut_26, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_27': xǁCompositeDiffusionModelǁpredict__mutmut_27, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_28': xǁCompositeDiffusionModelǁpredict__mutmut_28, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_29': xǁCompositeDiffusionModelǁpredict__mutmut_29, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_30': xǁCompositeDiffusionModelǁpredict__mutmut_30, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_31': xǁCompositeDiffusionModelǁpredict__mutmut_31, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_32': xǁCompositeDiffusionModelǁpredict__mutmut_32, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_33': xǁCompositeDiffusionModelǁpredict__mutmut_33, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_34': xǁCompositeDiffusionModelǁpredict__mutmut_34, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_35': xǁCompositeDiffusionModelǁpredict__mutmut_35, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_36': xǁCompositeDiffusionModelǁpredict__mutmut_36, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_37': xǁCompositeDiffusionModelǁpredict__mutmut_37, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_38': xǁCompositeDiffusionModelǁpredict__mutmut_38, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_39': xǁCompositeDiffusionModelǁpredict__mutmut_39, 
        'xǁCompositeDiffusionModelǁpredict__mutmut_40': xǁCompositeDiffusionModelǁpredict__mutmut_40
    }
    xǁCompositeDiffusionModelǁpredict__mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁpredict'

    def differential_equation(self, t, y, params):
        args = [t, y, params]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_orig(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_1(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = None
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_2(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(None)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_3(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = None

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_4(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = None
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_5(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 1
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_6(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = None
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_7(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = None
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_8(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(None)
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_9(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx - num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_10(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx = num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_11(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx -= num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_12(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = None

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_13(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = None

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_14(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros(None)

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_15(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = None
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_16(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 1
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_17(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(None):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_18(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(None):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_19(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i == j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_20(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = None
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_21(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx = 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_22(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx -= 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_23(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 2

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_24(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(None):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_25(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = None

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_26(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = None

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_27(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                None,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_28(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                None,
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_29(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                None,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_30(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                None,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_31(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_32(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_33(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_34(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_35(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_36(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i - 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_37(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 2],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_38(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = None

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_39(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(None)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_40(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] / y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_41(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(None) if i != j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_42(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i == j)

            dydt[i] = growth_rate - interaction_effect

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_43(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = None

        return dydt

    def xǁCompositeDiffusionModelǁdifferential_equation__mutmut_44(self, t, y, params):
        """Defines the composite diffusion model's differential equations."""
        dydt = B.zeros_like(y)
        param_list = [params[name] for name in self.param_names] if isinstance(params, dict) else params

        param_idx = 0
        model_params_list = []
        for model in self.models:
            num_params = len(model.param_names)
            model_params_list.append(param_list[param_idx : param_idx + num_params])
            param_idx += num_params

        alpha_params = param_list[param_idx:]

        alpha = B.zeros((self.n_models, self.n_models))

        alpha_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i != j:
                    alpha[i, j] = alpha_params[alpha_idx]
                    alpha_idx += 1

        for i, model in enumerate(self.models):
            model_params = model_params_list[i]

            # The differential_equation of the individual models is not directly used.
            # Instead, we call the differential_equation of the growth model.
            growth_rate = model.differential_equation(
                t,
                y[i : i + 1],
                model_params,
                None,
                t,
            )

            # Add interaction effects
            interaction_effect = sum(alpha[i, j] * y[j] for j in range(self.n_models) if i != j)

            dydt[i] = growth_rate + interaction_effect

        return dydt
    
    xǁCompositeDiffusionModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_1': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_1, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_2': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_2, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_3': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_3, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_4': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_4, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_5': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_5, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_6': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_6, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_7': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_7, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_8': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_8, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_9': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_9, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_10': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_10, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_11': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_11, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_12': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_12, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_13': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_13, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_14': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_14, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_15': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_15, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_16': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_16, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_17': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_17, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_18': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_18, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_19': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_19, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_20': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_20, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_21': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_21, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_22': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_22, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_23': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_23, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_24': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_24, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_25': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_25, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_26': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_26, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_27': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_27, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_28': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_28, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_29': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_29, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_30': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_30, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_31': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_31, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_32': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_32, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_33': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_33, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_34': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_34, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_35': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_35, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_36': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_36, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_37': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_37, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_38': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_38, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_39': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_39, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_40': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_40, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_41': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_41, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_42': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_42, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_43': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_43, 
        'xǁCompositeDiffusionModelǁdifferential_equation__mutmut_44': xǁCompositeDiffusionModelǁdifferential_equation__mutmut_44
    }
    xǁCompositeDiffusionModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁdifferential_equation'

    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁscore__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> float:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = None
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(None)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = None
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum(None)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) * 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y + y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 3)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = None
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum(None)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) * 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y + np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(None, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=None)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, )) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=1)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 3)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 + ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 2 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res * ss_tot if ss_tot > 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot >= 0 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 1 else 0.0

    def xǁCompositeDiffusionModelǁscore__mutmut_28(self, t: Sequence[float], y: Sequence[float]) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    
    xǁCompositeDiffusionModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁscore__mutmut_1': xǁCompositeDiffusionModelǁscore__mutmut_1, 
        'xǁCompositeDiffusionModelǁscore__mutmut_2': xǁCompositeDiffusionModelǁscore__mutmut_2, 
        'xǁCompositeDiffusionModelǁscore__mutmut_3': xǁCompositeDiffusionModelǁscore__mutmut_3, 
        'xǁCompositeDiffusionModelǁscore__mutmut_4': xǁCompositeDiffusionModelǁscore__mutmut_4, 
        'xǁCompositeDiffusionModelǁscore__mutmut_5': xǁCompositeDiffusionModelǁscore__mutmut_5, 
        'xǁCompositeDiffusionModelǁscore__mutmut_6': xǁCompositeDiffusionModelǁscore__mutmut_6, 
        'xǁCompositeDiffusionModelǁscore__mutmut_7': xǁCompositeDiffusionModelǁscore__mutmut_7, 
        'xǁCompositeDiffusionModelǁscore__mutmut_8': xǁCompositeDiffusionModelǁscore__mutmut_8, 
        'xǁCompositeDiffusionModelǁscore__mutmut_9': xǁCompositeDiffusionModelǁscore__mutmut_9, 
        'xǁCompositeDiffusionModelǁscore__mutmut_10': xǁCompositeDiffusionModelǁscore__mutmut_10, 
        'xǁCompositeDiffusionModelǁscore__mutmut_11': xǁCompositeDiffusionModelǁscore__mutmut_11, 
        'xǁCompositeDiffusionModelǁscore__mutmut_12': xǁCompositeDiffusionModelǁscore__mutmut_12, 
        'xǁCompositeDiffusionModelǁscore__mutmut_13': xǁCompositeDiffusionModelǁscore__mutmut_13, 
        'xǁCompositeDiffusionModelǁscore__mutmut_14': xǁCompositeDiffusionModelǁscore__mutmut_14, 
        'xǁCompositeDiffusionModelǁscore__mutmut_15': xǁCompositeDiffusionModelǁscore__mutmut_15, 
        'xǁCompositeDiffusionModelǁscore__mutmut_16': xǁCompositeDiffusionModelǁscore__mutmut_16, 
        'xǁCompositeDiffusionModelǁscore__mutmut_17': xǁCompositeDiffusionModelǁscore__mutmut_17, 
        'xǁCompositeDiffusionModelǁscore__mutmut_18': xǁCompositeDiffusionModelǁscore__mutmut_18, 
        'xǁCompositeDiffusionModelǁscore__mutmut_19': xǁCompositeDiffusionModelǁscore__mutmut_19, 
        'xǁCompositeDiffusionModelǁscore__mutmut_20': xǁCompositeDiffusionModelǁscore__mutmut_20, 
        'xǁCompositeDiffusionModelǁscore__mutmut_21': xǁCompositeDiffusionModelǁscore__mutmut_21, 
        'xǁCompositeDiffusionModelǁscore__mutmut_22': xǁCompositeDiffusionModelǁscore__mutmut_22, 
        'xǁCompositeDiffusionModelǁscore__mutmut_23': xǁCompositeDiffusionModelǁscore__mutmut_23, 
        'xǁCompositeDiffusionModelǁscore__mutmut_24': xǁCompositeDiffusionModelǁscore__mutmut_24, 
        'xǁCompositeDiffusionModelǁscore__mutmut_25': xǁCompositeDiffusionModelǁscore__mutmut_25, 
        'xǁCompositeDiffusionModelǁscore__mutmut_26': xǁCompositeDiffusionModelǁscore__mutmut_26, 
        'xǁCompositeDiffusionModelǁscore__mutmut_27': xǁCompositeDiffusionModelǁscore__mutmut_27, 
        'xǁCompositeDiffusionModelǁscore__mutmut_28': xǁCompositeDiffusionModelǁscore__mutmut_28
    }
    xǁCompositeDiffusionModelǁscore__mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁscore'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_orig(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_1(self, t: Sequence[float]) -> Sequence[float]:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_2(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError(None)

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_3(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_4(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_5(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_6(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = None
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_7(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(None)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_8(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = None
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_9(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            None,
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_10(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(None, yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_11(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, None, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_12(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, None) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_13(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(yi, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_14(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, self._params) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_15(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, ) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_16(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(None, y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_17(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, None)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_18(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(y_pred)],
        )
        return rates

    def xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_19(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t)
        rates = np.array(
            [self.differential_equation(ti, yi, self._params) for ti, yi in zip(t, )],
        )
        return rates
    
    xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_1': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_1, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_2': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_2, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_3': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_3, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_4': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_4, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_5': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_5, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_6': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_6, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_7': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_7, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_8': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_8, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_9': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_9, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_10': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_10, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_11': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_11, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_12': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_12, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_13': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_13, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_14': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_14, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_15': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_15, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_16': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_16, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_17': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_17, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_18': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_18, 
        'xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_19': xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_19
    }
    xǁCompositeDiffusionModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁCompositeDiffusionModelǁpredict_adoption_rate'
