# src/innovate/models/mixture.py

from collections.abc import Sequence

import numpy as np

from innovate.backend import current_backend as B
from innovate.base.base import DiffusionModel
from innovate.fitters.scipy_fitter import ScipyFitter
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


class MixtureModel(DiffusionModel):
    """A latent-class mixture model for diffusion dynamics.

    This model identifies distinct adopter segments from the data by fitting
    multiple diffusion submodels simultaneously. It uses the Expectation-
    Maximization (EM) algorithm to infer both the parameters of each submodel
    and the probability that each data point belongs to a particular segment.

    Parameters
    ----------
    model_classes : Sequence[Type[DiffusionModel]]
        A list of diffusion model classes (e.g., [Bass, Gompertz]) to use as
        the components of the mixture.
    max_iter : int, optional
        The maximum number of iterations for the EM algorithm (default is 100).
    tol : float, optional
        The tolerance for convergence of the log-likelihood (default is 1e-6).
    """

    def __init__(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        args = [models, weights, max_iter, tol]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁ__init____mutmut_orig(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_1(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 101,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_2(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1.000001,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_3(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_4(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError(None)

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_5(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("XXAt least one model is required.XX")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_6(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("at least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_7(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("AT LEAST ONE MODEL IS REQUIRED.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_8(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = None
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_9(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = None

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_10(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_11(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) == self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_12(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError(None)
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_13(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("XXNumber of weights must match number of models.XX")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_14(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_15(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("NUMBER OF WEIGHTS MUST MATCH NUMBER OF MODELS.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_16(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_17(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(None, 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_18(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), None):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_19(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_20(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), ):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_21(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(None), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_22(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 2.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_23(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError(None)
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_24(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("XXWeights must sum to 1.XX")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_25(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_26(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("WEIGHTS MUST SUM TO 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_27(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = None
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_28(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(None)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_29(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = None

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_30(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) * self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_31(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(None) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_32(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = None
        self.tol = tol
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_33(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = None
        self._params: dict[str, float] = {}

    def xǁMixtureModelǁ__init____mutmut_34(
        self,
        models: Sequence[DiffusionModel],
        weights: Sequence[float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.num_components = len(models)

        if weights is not None:
            if len(weights) != self.num_components:
                raise ValueError("Number of weights must match number of models.")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.")
            self.weights = B.array(weights)
        else:
            self.weights = B.ones(self.num_components) / self.num_components

        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[str, float] = None
    
    xǁMixtureModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁ__init____mutmut_1': xǁMixtureModelǁ__init____mutmut_1, 
        'xǁMixtureModelǁ__init____mutmut_2': xǁMixtureModelǁ__init____mutmut_2, 
        'xǁMixtureModelǁ__init____mutmut_3': xǁMixtureModelǁ__init____mutmut_3, 
        'xǁMixtureModelǁ__init____mutmut_4': xǁMixtureModelǁ__init____mutmut_4, 
        'xǁMixtureModelǁ__init____mutmut_5': xǁMixtureModelǁ__init____mutmut_5, 
        'xǁMixtureModelǁ__init____mutmut_6': xǁMixtureModelǁ__init____mutmut_6, 
        'xǁMixtureModelǁ__init____mutmut_7': xǁMixtureModelǁ__init____mutmut_7, 
        'xǁMixtureModelǁ__init____mutmut_8': xǁMixtureModelǁ__init____mutmut_8, 
        'xǁMixtureModelǁ__init____mutmut_9': xǁMixtureModelǁ__init____mutmut_9, 
        'xǁMixtureModelǁ__init____mutmut_10': xǁMixtureModelǁ__init____mutmut_10, 
        'xǁMixtureModelǁ__init____mutmut_11': xǁMixtureModelǁ__init____mutmut_11, 
        'xǁMixtureModelǁ__init____mutmut_12': xǁMixtureModelǁ__init____mutmut_12, 
        'xǁMixtureModelǁ__init____mutmut_13': xǁMixtureModelǁ__init____mutmut_13, 
        'xǁMixtureModelǁ__init____mutmut_14': xǁMixtureModelǁ__init____mutmut_14, 
        'xǁMixtureModelǁ__init____mutmut_15': xǁMixtureModelǁ__init____mutmut_15, 
        'xǁMixtureModelǁ__init____mutmut_16': xǁMixtureModelǁ__init____mutmut_16, 
        'xǁMixtureModelǁ__init____mutmut_17': xǁMixtureModelǁ__init____mutmut_17, 
        'xǁMixtureModelǁ__init____mutmut_18': xǁMixtureModelǁ__init____mutmut_18, 
        'xǁMixtureModelǁ__init____mutmut_19': xǁMixtureModelǁ__init____mutmut_19, 
        'xǁMixtureModelǁ__init____mutmut_20': xǁMixtureModelǁ__init____mutmut_20, 
        'xǁMixtureModelǁ__init____mutmut_21': xǁMixtureModelǁ__init____mutmut_21, 
        'xǁMixtureModelǁ__init____mutmut_22': xǁMixtureModelǁ__init____mutmut_22, 
        'xǁMixtureModelǁ__init____mutmut_23': xǁMixtureModelǁ__init____mutmut_23, 
        'xǁMixtureModelǁ__init____mutmut_24': xǁMixtureModelǁ__init____mutmut_24, 
        'xǁMixtureModelǁ__init____mutmut_25': xǁMixtureModelǁ__init____mutmut_25, 
        'xǁMixtureModelǁ__init____mutmut_26': xǁMixtureModelǁ__init____mutmut_26, 
        'xǁMixtureModelǁ__init____mutmut_27': xǁMixtureModelǁ__init____mutmut_27, 
        'xǁMixtureModelǁ__init____mutmut_28': xǁMixtureModelǁ__init____mutmut_28, 
        'xǁMixtureModelǁ__init____mutmut_29': xǁMixtureModelǁ__init____mutmut_29, 
        'xǁMixtureModelǁ__init____mutmut_30': xǁMixtureModelǁ__init____mutmut_30, 
        'xǁMixtureModelǁ__init____mutmut_31': xǁMixtureModelǁ__init____mutmut_31, 
        'xǁMixtureModelǁ__init____mutmut_32': xǁMixtureModelǁ__init____mutmut_32, 
        'xǁMixtureModelǁ__init____mutmut_33': xǁMixtureModelǁ__init____mutmut_33, 
        'xǁMixtureModelǁ__init____mutmut_34': xǁMixtureModelǁ__init____mutmut_34
    }
    xǁMixtureModelǁ__init____mutmut_orig.__name__ = 'xǁMixtureModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        """The names of the model parameters."""
        names: list[str] = []
        for i, model in enumerate(self.models):
            for pname in model.param_names:
                names.append(f"model_{i}_{pname}")
        for i in range(self.num_components):
            names.append(f"weight_{i}")
        return names

    def fit(self, t: Sequence[float], y: Sequence[float]):
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁfit__mutmut_orig(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_1(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = None
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_2(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(None)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_3(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = None

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_4(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(None)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_5(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = None
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_6(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(None, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_7(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, None, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_8(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, None)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_9(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_10(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_11(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, )

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_12(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = None

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_13(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = +np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_14(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(None):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_15(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = None
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_16(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack(None)
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_17(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(None) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_18(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(None)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_19(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = None

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_20(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) - B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_21(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(None) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_22(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds - 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_23(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1.000000001) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_24(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                None,
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_25(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = None
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_26(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds + B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_27(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(None, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_28(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=None)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_29(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_30(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, )
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_31(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=1)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_32(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = None

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_33(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(None)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_34(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = None

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_35(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(None, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_36(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=None)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_37(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_38(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, )

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_39(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=2)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_40(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(None):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_41(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = None  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_42(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] - 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_43(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1.000000001  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_44(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(None, t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_45(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], None, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_46(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, None, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_47(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=None)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_48(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_49(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_50(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_51(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, )
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_52(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = None
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_53(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(None)
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_54(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(None, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_55(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=None))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_56(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_57(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, ))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_58(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=1))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_59(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(None) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_60(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood + log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_61(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) <= self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_62(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                return
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def xǁMixtureModelǁfit__mutmut_63(self, t: Sequence[float], y: Sequence[float]):
        """Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)

        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(
                self.weights[:, None],
            )

            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9  # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = None

        self._update_params_from_models()
        return self
    
    xǁMixtureModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁfit__mutmut_1': xǁMixtureModelǁfit__mutmut_1, 
        'xǁMixtureModelǁfit__mutmut_2': xǁMixtureModelǁfit__mutmut_2, 
        'xǁMixtureModelǁfit__mutmut_3': xǁMixtureModelǁfit__mutmut_3, 
        'xǁMixtureModelǁfit__mutmut_4': xǁMixtureModelǁfit__mutmut_4, 
        'xǁMixtureModelǁfit__mutmut_5': xǁMixtureModelǁfit__mutmut_5, 
        'xǁMixtureModelǁfit__mutmut_6': xǁMixtureModelǁfit__mutmut_6, 
        'xǁMixtureModelǁfit__mutmut_7': xǁMixtureModelǁfit__mutmut_7, 
        'xǁMixtureModelǁfit__mutmut_8': xǁMixtureModelǁfit__mutmut_8, 
        'xǁMixtureModelǁfit__mutmut_9': xǁMixtureModelǁfit__mutmut_9, 
        'xǁMixtureModelǁfit__mutmut_10': xǁMixtureModelǁfit__mutmut_10, 
        'xǁMixtureModelǁfit__mutmut_11': xǁMixtureModelǁfit__mutmut_11, 
        'xǁMixtureModelǁfit__mutmut_12': xǁMixtureModelǁfit__mutmut_12, 
        'xǁMixtureModelǁfit__mutmut_13': xǁMixtureModelǁfit__mutmut_13, 
        'xǁMixtureModelǁfit__mutmut_14': xǁMixtureModelǁfit__mutmut_14, 
        'xǁMixtureModelǁfit__mutmut_15': xǁMixtureModelǁfit__mutmut_15, 
        'xǁMixtureModelǁfit__mutmut_16': xǁMixtureModelǁfit__mutmut_16, 
        'xǁMixtureModelǁfit__mutmut_17': xǁMixtureModelǁfit__mutmut_17, 
        'xǁMixtureModelǁfit__mutmut_18': xǁMixtureModelǁfit__mutmut_18, 
        'xǁMixtureModelǁfit__mutmut_19': xǁMixtureModelǁfit__mutmut_19, 
        'xǁMixtureModelǁfit__mutmut_20': xǁMixtureModelǁfit__mutmut_20, 
        'xǁMixtureModelǁfit__mutmut_21': xǁMixtureModelǁfit__mutmut_21, 
        'xǁMixtureModelǁfit__mutmut_22': xǁMixtureModelǁfit__mutmut_22, 
        'xǁMixtureModelǁfit__mutmut_23': xǁMixtureModelǁfit__mutmut_23, 
        'xǁMixtureModelǁfit__mutmut_24': xǁMixtureModelǁfit__mutmut_24, 
        'xǁMixtureModelǁfit__mutmut_25': xǁMixtureModelǁfit__mutmut_25, 
        'xǁMixtureModelǁfit__mutmut_26': xǁMixtureModelǁfit__mutmut_26, 
        'xǁMixtureModelǁfit__mutmut_27': xǁMixtureModelǁfit__mutmut_27, 
        'xǁMixtureModelǁfit__mutmut_28': xǁMixtureModelǁfit__mutmut_28, 
        'xǁMixtureModelǁfit__mutmut_29': xǁMixtureModelǁfit__mutmut_29, 
        'xǁMixtureModelǁfit__mutmut_30': xǁMixtureModelǁfit__mutmut_30, 
        'xǁMixtureModelǁfit__mutmut_31': xǁMixtureModelǁfit__mutmut_31, 
        'xǁMixtureModelǁfit__mutmut_32': xǁMixtureModelǁfit__mutmut_32, 
        'xǁMixtureModelǁfit__mutmut_33': xǁMixtureModelǁfit__mutmut_33, 
        'xǁMixtureModelǁfit__mutmut_34': xǁMixtureModelǁfit__mutmut_34, 
        'xǁMixtureModelǁfit__mutmut_35': xǁMixtureModelǁfit__mutmut_35, 
        'xǁMixtureModelǁfit__mutmut_36': xǁMixtureModelǁfit__mutmut_36, 
        'xǁMixtureModelǁfit__mutmut_37': xǁMixtureModelǁfit__mutmut_37, 
        'xǁMixtureModelǁfit__mutmut_38': xǁMixtureModelǁfit__mutmut_38, 
        'xǁMixtureModelǁfit__mutmut_39': xǁMixtureModelǁfit__mutmut_39, 
        'xǁMixtureModelǁfit__mutmut_40': xǁMixtureModelǁfit__mutmut_40, 
        'xǁMixtureModelǁfit__mutmut_41': xǁMixtureModelǁfit__mutmut_41, 
        'xǁMixtureModelǁfit__mutmut_42': xǁMixtureModelǁfit__mutmut_42, 
        'xǁMixtureModelǁfit__mutmut_43': xǁMixtureModelǁfit__mutmut_43, 
        'xǁMixtureModelǁfit__mutmut_44': xǁMixtureModelǁfit__mutmut_44, 
        'xǁMixtureModelǁfit__mutmut_45': xǁMixtureModelǁfit__mutmut_45, 
        'xǁMixtureModelǁfit__mutmut_46': xǁMixtureModelǁfit__mutmut_46, 
        'xǁMixtureModelǁfit__mutmut_47': xǁMixtureModelǁfit__mutmut_47, 
        'xǁMixtureModelǁfit__mutmut_48': xǁMixtureModelǁfit__mutmut_48, 
        'xǁMixtureModelǁfit__mutmut_49': xǁMixtureModelǁfit__mutmut_49, 
        'xǁMixtureModelǁfit__mutmut_50': xǁMixtureModelǁfit__mutmut_50, 
        'xǁMixtureModelǁfit__mutmut_51': xǁMixtureModelǁfit__mutmut_51, 
        'xǁMixtureModelǁfit__mutmut_52': xǁMixtureModelǁfit__mutmut_52, 
        'xǁMixtureModelǁfit__mutmut_53': xǁMixtureModelǁfit__mutmut_53, 
        'xǁMixtureModelǁfit__mutmut_54': xǁMixtureModelǁfit__mutmut_54, 
        'xǁMixtureModelǁfit__mutmut_55': xǁMixtureModelǁfit__mutmut_55, 
        'xǁMixtureModelǁfit__mutmut_56': xǁMixtureModelǁfit__mutmut_56, 
        'xǁMixtureModelǁfit__mutmut_57': xǁMixtureModelǁfit__mutmut_57, 
        'xǁMixtureModelǁfit__mutmut_58': xǁMixtureModelǁfit__mutmut_58, 
        'xǁMixtureModelǁfit__mutmut_59': xǁMixtureModelǁfit__mutmut_59, 
        'xǁMixtureModelǁfit__mutmut_60': xǁMixtureModelǁfit__mutmut_60, 
        'xǁMixtureModelǁfit__mutmut_61': xǁMixtureModelǁfit__mutmut_61, 
        'xǁMixtureModelǁfit__mutmut_62': xǁMixtureModelǁfit__mutmut_62, 
        'xǁMixtureModelǁfit__mutmut_63': xǁMixtureModelǁfit__mutmut_63
    }
    xǁMixtureModelǁfit__mutmut_orig.__name__ = 'xǁMixtureModelǁfit'

    def _update_params_from_models(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁ_update_params_from_models__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁ_update_params_from_models__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁ_update_params_from_models__mutmut_orig(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = {}
        for i, model in enumerate(self.models):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        for i, w in enumerate(self.weights):
            self._params[f"weight_{i}"] = w

    def xǁMixtureModelǁ_update_params_from_models__mutmut_1(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = None
        for i, model in enumerate(self.models):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        for i, w in enumerate(self.weights):
            self._params[f"weight_{i}"] = w

    def xǁMixtureModelǁ_update_params_from_models__mutmut_2(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = {}
        for i, model in enumerate(None):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        for i, w in enumerate(self.weights):
            self._params[f"weight_{i}"] = w

    def xǁMixtureModelǁ_update_params_from_models__mutmut_3(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = {}
        for i, model in enumerate(self.models):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = None
        for i, w in enumerate(self.weights):
            self._params[f"weight_{i}"] = w

    def xǁMixtureModelǁ_update_params_from_models__mutmut_4(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = {}
        for i, model in enumerate(self.models):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        for i, w in enumerate(None):
            self._params[f"weight_{i}"] = w

    def xǁMixtureModelǁ_update_params_from_models__mutmut_5(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = {}
        for i, model in enumerate(self.models):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        for i, w in enumerate(self.weights):
            self._params[f"weight_{i}"] = None
    
    xǁMixtureModelǁ_update_params_from_models__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁ_update_params_from_models__mutmut_1': xǁMixtureModelǁ_update_params_from_models__mutmut_1, 
        'xǁMixtureModelǁ_update_params_from_models__mutmut_2': xǁMixtureModelǁ_update_params_from_models__mutmut_2, 
        'xǁMixtureModelǁ_update_params_from_models__mutmut_3': xǁMixtureModelǁ_update_params_from_models__mutmut_3, 
        'xǁMixtureModelǁ_update_params_from_models__mutmut_4': xǁMixtureModelǁ_update_params_from_models__mutmut_4, 
        'xǁMixtureModelǁ_update_params_from_models__mutmut_5': xǁMixtureModelǁ_update_params_from_models__mutmut_5
    }
    xǁMixtureModelǁ_update_params_from_models__mutmut_orig.__name__ = 'xǁMixtureModelǁ_update_params_from_models'

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁpredict__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError(None)

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = None
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(None)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = None

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack(None)

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(None) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(None)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = None
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(None, axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=None)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], )
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds / self.weights[:, None], axis=0)
        return y_pred

    def xǁMixtureModelǁpredict__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """Makes predictions using the fitted mixture model."""
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])

        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=1)
        return y_pred
    
    xǁMixtureModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁpredict__mutmut_1': xǁMixtureModelǁpredict__mutmut_1, 
        'xǁMixtureModelǁpredict__mutmut_2': xǁMixtureModelǁpredict__mutmut_2, 
        'xǁMixtureModelǁpredict__mutmut_3': xǁMixtureModelǁpredict__mutmut_3, 
        'xǁMixtureModelǁpredict__mutmut_4': xǁMixtureModelǁpredict__mutmut_4, 
        'xǁMixtureModelǁpredict__mutmut_5': xǁMixtureModelǁpredict__mutmut_5, 
        'xǁMixtureModelǁpredict__mutmut_6': xǁMixtureModelǁpredict__mutmut_6, 
        'xǁMixtureModelǁpredict__mutmut_7': xǁMixtureModelǁpredict__mutmut_7, 
        'xǁMixtureModelǁpredict__mutmut_8': xǁMixtureModelǁpredict__mutmut_8, 
        'xǁMixtureModelǁpredict__mutmut_9': xǁMixtureModelǁpredict__mutmut_9, 
        'xǁMixtureModelǁpredict__mutmut_10': xǁMixtureModelǁpredict__mutmut_10, 
        'xǁMixtureModelǁpredict__mutmut_11': xǁMixtureModelǁpredict__mutmut_11, 
        'xǁMixtureModelǁpredict__mutmut_12': xǁMixtureModelǁpredict__mutmut_12, 
        'xǁMixtureModelǁpredict__mutmut_13': xǁMixtureModelǁpredict__mutmut_13, 
        'xǁMixtureModelǁpredict__mutmut_14': xǁMixtureModelǁpredict__mutmut_14, 
        'xǁMixtureModelǁpredict__mutmut_15': xǁMixtureModelǁpredict__mutmut_15, 
        'xǁMixtureModelǁpredict__mutmut_16': xǁMixtureModelǁpredict__mutmut_16, 
        'xǁMixtureModelǁpredict__mutmut_17': xǁMixtureModelǁpredict__mutmut_17, 
        'xǁMixtureModelǁpredict__mutmut_18': xǁMixtureModelǁpredict__mutmut_18
    }
    xǁMixtureModelǁpredict__mutmut_orig.__name__ = 'xǁMixtureModelǁpredict'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        """Sets the model parameters and updates the internal models."""
        self._params = value
        # Update weights
        self.weights = B.array(
            [value.get(f"weight_{i}", 0) for i in range(self.num_components)],
        )
        # Update submodel parameters
        for i, model in enumerate(self.models):
            prefix = f"model_{i}_"
            model_params = {k[len(prefix) :]: v for k, v in value.items() if k.startswith(prefix)}
            model.params_ = model_params

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        args = [t, y, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁscore__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = None
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(None, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, None)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, )
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = None
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum(None)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) * 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) + y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(None) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 3)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) + B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(None) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(None))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁMixtureModelǁscore__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁMixtureModelǁscore__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        """Calculates the R-squared score for the model."""
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁMixtureModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁscore__mutmut_1': xǁMixtureModelǁscore__mutmut_1, 
        'xǁMixtureModelǁscore__mutmut_2': xǁMixtureModelǁscore__mutmut_2, 
        'xǁMixtureModelǁscore__mutmut_3': xǁMixtureModelǁscore__mutmut_3, 
        'xǁMixtureModelǁscore__mutmut_4': xǁMixtureModelǁscore__mutmut_4, 
        'xǁMixtureModelǁscore__mutmut_5': xǁMixtureModelǁscore__mutmut_5, 
        'xǁMixtureModelǁscore__mutmut_6': xǁMixtureModelǁscore__mutmut_6, 
        'xǁMixtureModelǁscore__mutmut_7': xǁMixtureModelǁscore__mutmut_7, 
        'xǁMixtureModelǁscore__mutmut_8': xǁMixtureModelǁscore__mutmut_8, 
        'xǁMixtureModelǁscore__mutmut_9': xǁMixtureModelǁscore__mutmut_9, 
        'xǁMixtureModelǁscore__mutmut_10': xǁMixtureModelǁscore__mutmut_10, 
        'xǁMixtureModelǁscore__mutmut_11': xǁMixtureModelǁscore__mutmut_11, 
        'xǁMixtureModelǁscore__mutmut_12': xǁMixtureModelǁscore__mutmut_12, 
        'xǁMixtureModelǁscore__mutmut_13': xǁMixtureModelǁscore__mutmut_13, 
        'xǁMixtureModelǁscore__mutmut_14': xǁMixtureModelǁscore__mutmut_14, 
        'xǁMixtureModelǁscore__mutmut_15': xǁMixtureModelǁscore__mutmut_15, 
        'xǁMixtureModelǁscore__mutmut_16': xǁMixtureModelǁscore__mutmut_16, 
        'xǁMixtureModelǁscore__mutmut_17': xǁMixtureModelǁscore__mutmut_17, 
        'xǁMixtureModelǁscore__mutmut_18': xǁMixtureModelǁscore__mutmut_18, 
        'xǁMixtureModelǁscore__mutmut_19': xǁMixtureModelǁscore__mutmut_19, 
        'xǁMixtureModelǁscore__mutmut_20': xǁMixtureModelǁscore__mutmut_20, 
        'xǁMixtureModelǁscore__mutmut_21': xǁMixtureModelǁscore__mutmut_21, 
        'xǁMixtureModelǁscore__mutmut_22': xǁMixtureModelǁscore__mutmut_22, 
        'xǁMixtureModelǁscore__mutmut_23': xǁMixtureModelǁscore__mutmut_23, 
        'xǁMixtureModelǁscore__mutmut_24': xǁMixtureModelǁscore__mutmut_24, 
        'xǁMixtureModelǁscore__mutmut_25': xǁMixtureModelǁscore__mutmut_25
    }
    xǁMixtureModelǁscore__mutmut_orig.__name__ = 'xǁMixtureModelǁscore'

    def __repr__(self):
        return f"MixtureModel(models={self.model_classes}, weights={self.weights})"

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = None
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(None):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = None
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(None, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, None)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, )
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = None
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(None):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = None
        return bounds

    def xǁMixtureModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (1, 1)
        return bounds

    def xǁMixtureModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            bounds[f"weight_{i}"] = (0, 2)
        return bounds
    
    xǁMixtureModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁbounds__mutmut_1': xǁMixtureModelǁbounds__mutmut_1, 
        'xǁMixtureModelǁbounds__mutmut_2': xǁMixtureModelǁbounds__mutmut_2, 
        'xǁMixtureModelǁbounds__mutmut_3': xǁMixtureModelǁbounds__mutmut_3, 
        'xǁMixtureModelǁbounds__mutmut_4': xǁMixtureModelǁbounds__mutmut_4, 
        'xǁMixtureModelǁbounds__mutmut_5': xǁMixtureModelǁbounds__mutmut_5, 
        'xǁMixtureModelǁbounds__mutmut_6': xǁMixtureModelǁbounds__mutmut_6, 
        'xǁMixtureModelǁbounds__mutmut_7': xǁMixtureModelǁbounds__mutmut_7, 
        'xǁMixtureModelǁbounds__mutmut_8': xǁMixtureModelǁbounds__mutmut_8, 
        'xǁMixtureModelǁbounds__mutmut_9': xǁMixtureModelǁbounds__mutmut_9, 
        'xǁMixtureModelǁbounds__mutmut_10': xǁMixtureModelǁbounds__mutmut_10, 
        'xǁMixtureModelǁbounds__mutmut_11': xǁMixtureModelǁbounds__mutmut_11, 
        'xǁMixtureModelǁbounds__mutmut_12': xǁMixtureModelǁbounds__mutmut_12
    }
    xǁMixtureModelǁbounds__mutmut_orig.__name__ = 'xǁMixtureModelǁbounds'

    def differential_equation(self, y, t, p):
        pass

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = None
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(None):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = None
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(None, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, None)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, )
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = None
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(None):
            guesses[f"weight_{i}"] = 1 / self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = None
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 1 * self.num_components
        return guesses

    def xǁMixtureModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i in range(self.num_components):
            guesses[f"weight_{i}"] = 2 / self.num_components
        return guesses
    
    xǁMixtureModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁinitial_guesses__mutmut_1': xǁMixtureModelǁinitial_guesses__mutmut_1, 
        'xǁMixtureModelǁinitial_guesses__mutmut_2': xǁMixtureModelǁinitial_guesses__mutmut_2, 
        'xǁMixtureModelǁinitial_guesses__mutmut_3': xǁMixtureModelǁinitial_guesses__mutmut_3, 
        'xǁMixtureModelǁinitial_guesses__mutmut_4': xǁMixtureModelǁinitial_guesses__mutmut_4, 
        'xǁMixtureModelǁinitial_guesses__mutmut_5': xǁMixtureModelǁinitial_guesses__mutmut_5, 
        'xǁMixtureModelǁinitial_guesses__mutmut_6': xǁMixtureModelǁinitial_guesses__mutmut_6, 
        'xǁMixtureModelǁinitial_guesses__mutmut_7': xǁMixtureModelǁinitial_guesses__mutmut_7, 
        'xǁMixtureModelǁinitial_guesses__mutmut_8': xǁMixtureModelǁinitial_guesses__mutmut_8, 
        'xǁMixtureModelǁinitial_guesses__mutmut_9': xǁMixtureModelǁinitial_guesses__mutmut_9, 
        'xǁMixtureModelǁinitial_guesses__mutmut_10': xǁMixtureModelǁinitial_guesses__mutmut_10, 
        'xǁMixtureModelǁinitial_guesses__mutmut_11': xǁMixtureModelǁinitial_guesses__mutmut_11, 
        'xǁMixtureModelǁinitial_guesses__mutmut_12': xǁMixtureModelǁinitial_guesses__mutmut_12
    }
    xǁMixtureModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁMixtureModelǁinitial_guesses'

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMixtureModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁMixtureModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_orig(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_1(self, t: Sequence[float]) -> Sequence[float]:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_2(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError(None)

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_3(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_4(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_5(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_6(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = None
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_7(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(None)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_8(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = None

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_9(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            None,
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_10(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(None) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_11(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(None)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_12(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = None
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_13(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(None, axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_14(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=None)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_15(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_16(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], )
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_17(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates / self.weights[:, None], axis=0)
        return y_rate

    def xǁMixtureModelǁpredict_adoption_rate__mutmut_18(self, t: Sequence[float]) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = B.array(t)
        component_rates = B.stack(
            [B.array(m.predict_adoption_rate(t_arr)) for m in self.models],
        )

        # Weighted average of the component predictions
        y_rate = B.sum(component_rates * self.weights[:, None], axis=1)
        return y_rate
    
    xǁMixtureModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMixtureModelǁpredict_adoption_rate__mutmut_1': xǁMixtureModelǁpredict_adoption_rate__mutmut_1, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_2': xǁMixtureModelǁpredict_adoption_rate__mutmut_2, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_3': xǁMixtureModelǁpredict_adoption_rate__mutmut_3, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_4': xǁMixtureModelǁpredict_adoption_rate__mutmut_4, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_5': xǁMixtureModelǁpredict_adoption_rate__mutmut_5, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_6': xǁMixtureModelǁpredict_adoption_rate__mutmut_6, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_7': xǁMixtureModelǁpredict_adoption_rate__mutmut_7, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_8': xǁMixtureModelǁpredict_adoption_rate__mutmut_8, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_9': xǁMixtureModelǁpredict_adoption_rate__mutmut_9, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_10': xǁMixtureModelǁpredict_adoption_rate__mutmut_10, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_11': xǁMixtureModelǁpredict_adoption_rate__mutmut_11, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_12': xǁMixtureModelǁpredict_adoption_rate__mutmut_12, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_13': xǁMixtureModelǁpredict_adoption_rate__mutmut_13, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_14': xǁMixtureModelǁpredict_adoption_rate__mutmut_14, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_15': xǁMixtureModelǁpredict_adoption_rate__mutmut_15, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_16': xǁMixtureModelǁpredict_adoption_rate__mutmut_16, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_17': xǁMixtureModelǁpredict_adoption_rate__mutmut_17, 
        'xǁMixtureModelǁpredict_adoption_rate__mutmut_18': xǁMixtureModelǁpredict_adoption_rate__mutmut_18
    }
    xǁMixtureModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁMixtureModelǁpredict_adoption_rate'
