from collections.abc import Sequence

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


class BatchedFitter:
    """A fitter class for fitting a model to multiple datasets in a batch."""

    def __init__(self, model: DiffusionModel, fitter):
        args = [model, fitter]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBatchedFitterǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁBatchedFitterǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁBatchedFitterǁ__init____mutmut_orig(self, model: DiffusionModel, fitter):
        self.model = model
        self.fitter = fitter
        self.fitted_params = None

    def xǁBatchedFitterǁ__init____mutmut_1(self, model: DiffusionModel, fitter):
        self.model = None
        self.fitter = fitter
        self.fitted_params = None

    def xǁBatchedFitterǁ__init____mutmut_2(self, model: DiffusionModel, fitter):
        self.model = model
        self.fitter = None
        self.fitted_params = None

    def xǁBatchedFitterǁ__init____mutmut_3(self, model: DiffusionModel, fitter):
        self.model = model
        self.fitter = fitter
        self.fitted_params = ""
    
    xǁBatchedFitterǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBatchedFitterǁ__init____mutmut_1': xǁBatchedFitterǁ__init____mutmut_1, 
        'xǁBatchedFitterǁ__init____mutmut_2': xǁBatchedFitterǁ__init____mutmut_2, 
        'xǁBatchedFitterǁ__init____mutmut_3': xǁBatchedFitterǁ__init____mutmut_3
    }
    xǁBatchedFitterǁ__init____mutmut_orig.__name__ = 'xǁBatchedFitterǁ__init__'

    def fit(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        args = [t_batched, y_batched]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBatchedFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁBatchedFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁBatchedFitterǁfit__mutmut_orig(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_1(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) == len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_2(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                None,
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_3(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "XXThe number of time sequences and adoption sequences must be the same.XX",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_4(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "the number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_5(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "THE NUMBER OF TIME SEQUENCES AND ADOPTION SEQUENCES MUST BE THE SAME.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_6(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = None
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_7(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(None, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_8(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, None):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_9(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_10(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, ):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_11(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = None
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_12(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(None)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_13(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = None
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_14(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(None)
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_15(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(None, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_16(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, None).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_17(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_18(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, ).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_19(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = None
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_20(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(None)
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_21(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(None, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_22(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, None).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_23(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_24(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, ).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_25(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(None, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_26(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, None, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_27(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, None, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_28(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=None, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_29(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=None)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_30(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_31(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_32(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_33(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_34(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, )
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_35(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(None)

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_36(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(None))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_37(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = None
        return self.fitted_params

    def xǁBatchedFitterǁfit__mutmut_38(
        self,
        t_batched: Sequence[Sequence[float]],
        y_batched: Sequence[Sequence[float]],
    ):
        """Fits the model to a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same.",
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(None)
        return self.fitted_params
    
    xǁBatchedFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBatchedFitterǁfit__mutmut_1': xǁBatchedFitterǁfit__mutmut_1, 
        'xǁBatchedFitterǁfit__mutmut_2': xǁBatchedFitterǁfit__mutmut_2, 
        'xǁBatchedFitterǁfit__mutmut_3': xǁBatchedFitterǁfit__mutmut_3, 
        'xǁBatchedFitterǁfit__mutmut_4': xǁBatchedFitterǁfit__mutmut_4, 
        'xǁBatchedFitterǁfit__mutmut_5': xǁBatchedFitterǁfit__mutmut_5, 
        'xǁBatchedFitterǁfit__mutmut_6': xǁBatchedFitterǁfit__mutmut_6, 
        'xǁBatchedFitterǁfit__mutmut_7': xǁBatchedFitterǁfit__mutmut_7, 
        'xǁBatchedFitterǁfit__mutmut_8': xǁBatchedFitterǁfit__mutmut_8, 
        'xǁBatchedFitterǁfit__mutmut_9': xǁBatchedFitterǁfit__mutmut_9, 
        'xǁBatchedFitterǁfit__mutmut_10': xǁBatchedFitterǁfit__mutmut_10, 
        'xǁBatchedFitterǁfit__mutmut_11': xǁBatchedFitterǁfit__mutmut_11, 
        'xǁBatchedFitterǁfit__mutmut_12': xǁBatchedFitterǁfit__mutmut_12, 
        'xǁBatchedFitterǁfit__mutmut_13': xǁBatchedFitterǁfit__mutmut_13, 
        'xǁBatchedFitterǁfit__mutmut_14': xǁBatchedFitterǁfit__mutmut_14, 
        'xǁBatchedFitterǁfit__mutmut_15': xǁBatchedFitterǁfit__mutmut_15, 
        'xǁBatchedFitterǁfit__mutmut_16': xǁBatchedFitterǁfit__mutmut_16, 
        'xǁBatchedFitterǁfit__mutmut_17': xǁBatchedFitterǁfit__mutmut_17, 
        'xǁBatchedFitterǁfit__mutmut_18': xǁBatchedFitterǁfit__mutmut_18, 
        'xǁBatchedFitterǁfit__mutmut_19': xǁBatchedFitterǁfit__mutmut_19, 
        'xǁBatchedFitterǁfit__mutmut_20': xǁBatchedFitterǁfit__mutmut_20, 
        'xǁBatchedFitterǁfit__mutmut_21': xǁBatchedFitterǁfit__mutmut_21, 
        'xǁBatchedFitterǁfit__mutmut_22': xǁBatchedFitterǁfit__mutmut_22, 
        'xǁBatchedFitterǁfit__mutmut_23': xǁBatchedFitterǁfit__mutmut_23, 
        'xǁBatchedFitterǁfit__mutmut_24': xǁBatchedFitterǁfit__mutmut_24, 
        'xǁBatchedFitterǁfit__mutmut_25': xǁBatchedFitterǁfit__mutmut_25, 
        'xǁBatchedFitterǁfit__mutmut_26': xǁBatchedFitterǁfit__mutmut_26, 
        'xǁBatchedFitterǁfit__mutmut_27': xǁBatchedFitterǁfit__mutmut_27, 
        'xǁBatchedFitterǁfit__mutmut_28': xǁBatchedFitterǁfit__mutmut_28, 
        'xǁBatchedFitterǁfit__mutmut_29': xǁBatchedFitterǁfit__mutmut_29, 
        'xǁBatchedFitterǁfit__mutmut_30': xǁBatchedFitterǁfit__mutmut_30, 
        'xǁBatchedFitterǁfit__mutmut_31': xǁBatchedFitterǁfit__mutmut_31, 
        'xǁBatchedFitterǁfit__mutmut_32': xǁBatchedFitterǁfit__mutmut_32, 
        'xǁBatchedFitterǁfit__mutmut_33': xǁBatchedFitterǁfit__mutmut_33, 
        'xǁBatchedFitterǁfit__mutmut_34': xǁBatchedFitterǁfit__mutmut_34, 
        'xǁBatchedFitterǁfit__mutmut_35': xǁBatchedFitterǁfit__mutmut_35, 
        'xǁBatchedFitterǁfit__mutmut_36': xǁBatchedFitterǁfit__mutmut_36, 
        'xǁBatchedFitterǁfit__mutmut_37': xǁBatchedFitterǁfit__mutmut_37, 
        'xǁBatchedFitterǁfit__mutmut_38': xǁBatchedFitterǁfit__mutmut_38
    }
    xǁBatchedFitterǁfit__mutmut_orig.__name__ = 'xǁBatchedFitterǁfit'

    def predict(self, t_batched: Sequence[Sequence[float]]):
        args = [t_batched]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBatchedFitterǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁBatchedFitterǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁBatchedFitterǁpredict__mutmut_orig(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_1(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is not None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_2(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError(None)

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_3(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_4(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_5(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_6(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = None
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_7(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(None)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_8(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = None
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_9(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(None)
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_10(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(None, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_11(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, None))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_12(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_13(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, ))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_14(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = None
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_15(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(None)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_16(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = None
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_17(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(None)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_18(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = None
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_19(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(None, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_20(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, None)
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_21(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_22(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, )
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_23(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(None))
        return predictions.reshape(predictions.shape[0], -1)

    def xǁBatchedFitterǁpredict__mutmut_24(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(None, -1)

    def xǁBatchedFitterǁpredict__mutmut_25(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], None)

    def xǁBatchedFitterǁpredict__mutmut_26(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(-1)

    def xǁBatchedFitterǁpredict__mutmut_27(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], )

    def xǁBatchedFitterǁpredict__mutmut_28(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[1], -1)

    def xǁBatchedFitterǁpredict__mutmut_29(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], +1)

    def xǁBatchedFitterǁpredict__mutmut_30(self, t_batched: Sequence[Sequence[float]]):
        """Makes predictions for a batch of datasets.

        Args:
        ----
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -2)
    
    xǁBatchedFitterǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBatchedFitterǁpredict__mutmut_1': xǁBatchedFitterǁpredict__mutmut_1, 
        'xǁBatchedFitterǁpredict__mutmut_2': xǁBatchedFitterǁpredict__mutmut_2, 
        'xǁBatchedFitterǁpredict__mutmut_3': xǁBatchedFitterǁpredict__mutmut_3, 
        'xǁBatchedFitterǁpredict__mutmut_4': xǁBatchedFitterǁpredict__mutmut_4, 
        'xǁBatchedFitterǁpredict__mutmut_5': xǁBatchedFitterǁpredict__mutmut_5, 
        'xǁBatchedFitterǁpredict__mutmut_6': xǁBatchedFitterǁpredict__mutmut_6, 
        'xǁBatchedFitterǁpredict__mutmut_7': xǁBatchedFitterǁpredict__mutmut_7, 
        'xǁBatchedFitterǁpredict__mutmut_8': xǁBatchedFitterǁpredict__mutmut_8, 
        'xǁBatchedFitterǁpredict__mutmut_9': xǁBatchedFitterǁpredict__mutmut_9, 
        'xǁBatchedFitterǁpredict__mutmut_10': xǁBatchedFitterǁpredict__mutmut_10, 
        'xǁBatchedFitterǁpredict__mutmut_11': xǁBatchedFitterǁpredict__mutmut_11, 
        'xǁBatchedFitterǁpredict__mutmut_12': xǁBatchedFitterǁpredict__mutmut_12, 
        'xǁBatchedFitterǁpredict__mutmut_13': xǁBatchedFitterǁpredict__mutmut_13, 
        'xǁBatchedFitterǁpredict__mutmut_14': xǁBatchedFitterǁpredict__mutmut_14, 
        'xǁBatchedFitterǁpredict__mutmut_15': xǁBatchedFitterǁpredict__mutmut_15, 
        'xǁBatchedFitterǁpredict__mutmut_16': xǁBatchedFitterǁpredict__mutmut_16, 
        'xǁBatchedFitterǁpredict__mutmut_17': xǁBatchedFitterǁpredict__mutmut_17, 
        'xǁBatchedFitterǁpredict__mutmut_18': xǁBatchedFitterǁpredict__mutmut_18, 
        'xǁBatchedFitterǁpredict__mutmut_19': xǁBatchedFitterǁpredict__mutmut_19, 
        'xǁBatchedFitterǁpredict__mutmut_20': xǁBatchedFitterǁpredict__mutmut_20, 
        'xǁBatchedFitterǁpredict__mutmut_21': xǁBatchedFitterǁpredict__mutmut_21, 
        'xǁBatchedFitterǁpredict__mutmut_22': xǁBatchedFitterǁpredict__mutmut_22, 
        'xǁBatchedFitterǁpredict__mutmut_23': xǁBatchedFitterǁpredict__mutmut_23, 
        'xǁBatchedFitterǁpredict__mutmut_24': xǁBatchedFitterǁpredict__mutmut_24, 
        'xǁBatchedFitterǁpredict__mutmut_25': xǁBatchedFitterǁpredict__mutmut_25, 
        'xǁBatchedFitterǁpredict__mutmut_26': xǁBatchedFitterǁpredict__mutmut_26, 
        'xǁBatchedFitterǁpredict__mutmut_27': xǁBatchedFitterǁpredict__mutmut_27, 
        'xǁBatchedFitterǁpredict__mutmut_28': xǁBatchedFitterǁpredict__mutmut_28, 
        'xǁBatchedFitterǁpredict__mutmut_29': xǁBatchedFitterǁpredict__mutmut_29, 
        'xǁBatchedFitterǁpredict__mutmut_30': xǁBatchedFitterǁpredict__mutmut_30
    }
    xǁBatchedFitterǁpredict__mutmut_orig.__name__ = 'xǁBatchedFitterǁpredict'
