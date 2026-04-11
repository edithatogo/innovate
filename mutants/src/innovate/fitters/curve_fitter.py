# src/innovate/fitters/curve_fitter.py

import numpy as np
from scipy.optimize import curve_fit

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


class CurveFitter:
    """A fitter that uses scipy.optimize.curve_fit to estimate model parameters."""

    def __init__(self, model: DiffusionModel):
        args = [model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCurveFitterǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁCurveFitterǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁCurveFitterǁ__init____mutmut_orig(self, model: DiffusionModel):
        self.model = model

    def xǁCurveFitterǁ__init____mutmut_1(self, model: DiffusionModel):
        self.model = None
    
    xǁCurveFitterǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCurveFitterǁ__init____mutmut_1': xǁCurveFitterǁ__init____mutmut_1
    }
    xǁCurveFitterǁ__init____mutmut_orig.__name__ = 'xǁCurveFitterǁ__init__'

    def fit(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        args = [model, t, y, p0, bounds]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁCurveFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁCurveFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁCurveFitterǁfit__mutmut_orig(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_1(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = None
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_2(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = None
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_3(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(None)
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_4(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(None, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_5(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, None))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_6(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_7(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, ))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_8(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(None)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_9(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = None

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_10(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(None, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_11(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, None, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_12(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, None, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_13(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=None, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_14(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=None)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_15(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_16(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_17(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_18(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_19(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, )

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_20(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = None
        return self

    def xǁCurveFitterǁfit__mutmut_21(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(None)
        return self

    def xǁCurveFitterǁfit__mutmut_22(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(None, popt))
        return self

    def xǁCurveFitterǁfit__mutmut_23(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, None))
        return self

    def xǁCurveFitterǁfit__mutmut_24(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(popt))
        return self

    def xǁCurveFitterǁfit__mutmut_25(
        self,
        model: DiffusionModel,
        t: np.ndarray,
        y: np.ndarray,
        p0: list,
        bounds: tuple,
        **kwargs,
    ):
        """Fits the model to the data using curve_fit."""

        def func(t, *params):
            # The model's differential_equation is not directly used by curve_fit.
            # Instead, we need a function that returns the predicted y values.
            # We'll use a simplified version of the predict method for this.

            # Create a temporary model to set the parameters for prediction
            temp_model = self.model.__class__()
            temp_model.params_ = dict(zip(self.model.param_names, params))
            return temp_model.predict(t)

        # Use the model's initial guesses and bounds
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds)

        # Set the model parameters to the optimal values
        self.model.params_ = dict(zip(self.model.param_names, ))
        return self
    
    xǁCurveFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁCurveFitterǁfit__mutmut_1': xǁCurveFitterǁfit__mutmut_1, 
        'xǁCurveFitterǁfit__mutmut_2': xǁCurveFitterǁfit__mutmut_2, 
        'xǁCurveFitterǁfit__mutmut_3': xǁCurveFitterǁfit__mutmut_3, 
        'xǁCurveFitterǁfit__mutmut_4': xǁCurveFitterǁfit__mutmut_4, 
        'xǁCurveFitterǁfit__mutmut_5': xǁCurveFitterǁfit__mutmut_5, 
        'xǁCurveFitterǁfit__mutmut_6': xǁCurveFitterǁfit__mutmut_6, 
        'xǁCurveFitterǁfit__mutmut_7': xǁCurveFitterǁfit__mutmut_7, 
        'xǁCurveFitterǁfit__mutmut_8': xǁCurveFitterǁfit__mutmut_8, 
        'xǁCurveFitterǁfit__mutmut_9': xǁCurveFitterǁfit__mutmut_9, 
        'xǁCurveFitterǁfit__mutmut_10': xǁCurveFitterǁfit__mutmut_10, 
        'xǁCurveFitterǁfit__mutmut_11': xǁCurveFitterǁfit__mutmut_11, 
        'xǁCurveFitterǁfit__mutmut_12': xǁCurveFitterǁfit__mutmut_12, 
        'xǁCurveFitterǁfit__mutmut_13': xǁCurveFitterǁfit__mutmut_13, 
        'xǁCurveFitterǁfit__mutmut_14': xǁCurveFitterǁfit__mutmut_14, 
        'xǁCurveFitterǁfit__mutmut_15': xǁCurveFitterǁfit__mutmut_15, 
        'xǁCurveFitterǁfit__mutmut_16': xǁCurveFitterǁfit__mutmut_16, 
        'xǁCurveFitterǁfit__mutmut_17': xǁCurveFitterǁfit__mutmut_17, 
        'xǁCurveFitterǁfit__mutmut_18': xǁCurveFitterǁfit__mutmut_18, 
        'xǁCurveFitterǁfit__mutmut_19': xǁCurveFitterǁfit__mutmut_19, 
        'xǁCurveFitterǁfit__mutmut_20': xǁCurveFitterǁfit__mutmut_20, 
        'xǁCurveFitterǁfit__mutmut_21': xǁCurveFitterǁfit__mutmut_21, 
        'xǁCurveFitterǁfit__mutmut_22': xǁCurveFitterǁfit__mutmut_22, 
        'xǁCurveFitterǁfit__mutmut_23': xǁCurveFitterǁfit__mutmut_23, 
        'xǁCurveFitterǁfit__mutmut_24': xǁCurveFitterǁfit__mutmut_24, 
        'xǁCurveFitterǁfit__mutmut_25': xǁCurveFitterǁfit__mutmut_25
    }
    xǁCurveFitterǁfit__mutmut_orig.__name__ = 'xǁCurveFitterǁfit'
