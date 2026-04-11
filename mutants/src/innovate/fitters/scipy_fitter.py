from collections.abc import Sequence

import numpy as np
from scipy.optimize import curve_fit
from typing_extensions import Self

from innovate.base.base import DiffusionModel
from innovate.compete.competition import MultiProductDiffusionModel  # Import the model
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


class ScipyFitter:
    """A fitter class that uses SciPy's curve_fit for model estimation."""

    def fit(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        args = [model, t, y, p0, bounds, weights]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁScipyFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁScipyFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁScipyFitterǁfit__mutmut_orig(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_1(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = None
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_2(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(None)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_3(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = None
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_4(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(None)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_5(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_6(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 * np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_7(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 2.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_8(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(None) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_9(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_10(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_11(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    None,
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_12(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    None,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_13(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_14(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_15(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "XXMultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.XX",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_16(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "multiproductdiffusionmodel does not support sample weights. weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_17(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MULTIPRODUCTDIFFUSIONMODEL DOES NOT SUPPORT SAMPLE WEIGHTS. WEIGHTS PARAMETER WILL BE IGNORED.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_18(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_19(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = None

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_20(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["XXboundsXX"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_21(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["BOUNDS"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_22(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(None, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_23(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, None, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_24(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_25(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_26(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, )
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_27(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = None

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_28(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = None
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_29(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(None)
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_30(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(None, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_31(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, None))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_32(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_33(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, ))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_34(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = None
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_35(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(None).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_36(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = None

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_37(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is not None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_38(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = None

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_39(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(None)

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_40(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(None, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_41(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, None).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_42(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_43(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, ).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_44(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is not None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_45(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = None
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_46(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[1] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_47(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(None, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_48(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, None).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_49(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_50(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, ).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_51(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = None
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_52(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[2] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_53(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(None, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_54(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, None).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_55(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_56(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, ).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_57(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = None

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_58(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = None
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_59(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                None,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_60(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                None,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_61(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                None,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_62(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=None,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_63(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=None,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_64(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=None,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_65(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=None,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_66(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_67(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_68(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_69(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_70(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_71(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_72(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_73(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_74(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=False,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_75(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = None
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_76(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(None)
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_77(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(None, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_78(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, None))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_79(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_80(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, ))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_81(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(None)
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")

        return self

    def xǁScipyFitterǁfit__mutmut_82(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        p0: Sequence[float] | None = None,
        bounds: tuple | None = None,
        weights: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Fits a DiffusionModel instance using scipy.optimize.curve_fit.

        Args:
        ----
            model: An instance of a DiffusionModel (e.g., BassModel, GompertzModel, LogisticModel).
            t: Time points (independent variable).
            y: Observed adoption data (dependent variable).
            p0: Initial guesses for the parameters. If None, model.initial_guesses() is used.
            bounds: Bounds for the parameters. If None, model.bounds() is used.
            weights: Weights for the observed data points.
            kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

        Returns
        -------
            The fitter instance.

        Raises
        ------
            RuntimeError: If fitting fails.
        """
        t_arr = np.array(t)
        y_arr = np.array(y)
        sigma = 1.0 / np.sqrt(weights) if weights is not None else None

        # Check for MultiProductDiffusionModel and handle accordingly
        if isinstance(model, MultiProductDiffusionModel):
            # MultiProductDiffusionModel has its own sophisticated fitting method
            # using scipy.optimize.minimize which is more appropriate for multi-output models
            # than curve_fit. We delegate to the model's built-in fitting capability.
            if weights is not None:
                # Note: MultiProductDiffusionModel.fit() doesn't support weights parameter
                # This is a limitation we acknowledge
                import warnings

                warnings.warn(
                    "MultiProductDiffusionModel does not support sample weights. Weights parameter will be ignored.",
                    UserWarning,
                )

            # Convert bounds format if provided
            if bounds is not None:
                # Convert from curve_fit format to minimize format if needed
                # This is a simplified conversion - full conversion would require
                # understanding the parameter structure
                kwargs["bounds"] = bounds

            # Use the model's built-in fitting method
            model.fit(t, y, **kwargs)
            return self

        # Handle regular DiffusionModel instances with curve_fit
        y_arr = y_arr.flatten()

        def fit_function(t, *params):
            param_dict = dict(zip(model.param_names, params))
            model.params_ = param_dict
            return model.predict(t).flatten()

        x_fit = t_arr

        # Determine initial guesses if not provided
        if p0 is None:
            p0 = list(model.initial_guesses(t, y).values())

        # Determine bounds if not provided
        if bounds is None:
            lower_bounds = [b[0] for b in model.bounds(t, y).values()]
            upper_bounds = [b[1] for b in model.bounds(t, y).values()]
            bounds = (lower_bounds, upper_bounds)

        try:
            popt, _ = curve_fit(
                fit_function,
                x_fit,
                y_arr,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                **kwargs,
            )
            model.params_ = dict(zip(model.param_names, popt))
        except ValueError as e:
            raise RuntimeError(f"Fitting failed due to invalid parameters or data: {e}")
        except RuntimeError as e:
            raise RuntimeError(None)

        return self
    
    xǁScipyFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁScipyFitterǁfit__mutmut_1': xǁScipyFitterǁfit__mutmut_1, 
        'xǁScipyFitterǁfit__mutmut_2': xǁScipyFitterǁfit__mutmut_2, 
        'xǁScipyFitterǁfit__mutmut_3': xǁScipyFitterǁfit__mutmut_3, 
        'xǁScipyFitterǁfit__mutmut_4': xǁScipyFitterǁfit__mutmut_4, 
        'xǁScipyFitterǁfit__mutmut_5': xǁScipyFitterǁfit__mutmut_5, 
        'xǁScipyFitterǁfit__mutmut_6': xǁScipyFitterǁfit__mutmut_6, 
        'xǁScipyFitterǁfit__mutmut_7': xǁScipyFitterǁfit__mutmut_7, 
        'xǁScipyFitterǁfit__mutmut_8': xǁScipyFitterǁfit__mutmut_8, 
        'xǁScipyFitterǁfit__mutmut_9': xǁScipyFitterǁfit__mutmut_9, 
        'xǁScipyFitterǁfit__mutmut_10': xǁScipyFitterǁfit__mutmut_10, 
        'xǁScipyFitterǁfit__mutmut_11': xǁScipyFitterǁfit__mutmut_11, 
        'xǁScipyFitterǁfit__mutmut_12': xǁScipyFitterǁfit__mutmut_12, 
        'xǁScipyFitterǁfit__mutmut_13': xǁScipyFitterǁfit__mutmut_13, 
        'xǁScipyFitterǁfit__mutmut_14': xǁScipyFitterǁfit__mutmut_14, 
        'xǁScipyFitterǁfit__mutmut_15': xǁScipyFitterǁfit__mutmut_15, 
        'xǁScipyFitterǁfit__mutmut_16': xǁScipyFitterǁfit__mutmut_16, 
        'xǁScipyFitterǁfit__mutmut_17': xǁScipyFitterǁfit__mutmut_17, 
        'xǁScipyFitterǁfit__mutmut_18': xǁScipyFitterǁfit__mutmut_18, 
        'xǁScipyFitterǁfit__mutmut_19': xǁScipyFitterǁfit__mutmut_19, 
        'xǁScipyFitterǁfit__mutmut_20': xǁScipyFitterǁfit__mutmut_20, 
        'xǁScipyFitterǁfit__mutmut_21': xǁScipyFitterǁfit__mutmut_21, 
        'xǁScipyFitterǁfit__mutmut_22': xǁScipyFitterǁfit__mutmut_22, 
        'xǁScipyFitterǁfit__mutmut_23': xǁScipyFitterǁfit__mutmut_23, 
        'xǁScipyFitterǁfit__mutmut_24': xǁScipyFitterǁfit__mutmut_24, 
        'xǁScipyFitterǁfit__mutmut_25': xǁScipyFitterǁfit__mutmut_25, 
        'xǁScipyFitterǁfit__mutmut_26': xǁScipyFitterǁfit__mutmut_26, 
        'xǁScipyFitterǁfit__mutmut_27': xǁScipyFitterǁfit__mutmut_27, 
        'xǁScipyFitterǁfit__mutmut_28': xǁScipyFitterǁfit__mutmut_28, 
        'xǁScipyFitterǁfit__mutmut_29': xǁScipyFitterǁfit__mutmut_29, 
        'xǁScipyFitterǁfit__mutmut_30': xǁScipyFitterǁfit__mutmut_30, 
        'xǁScipyFitterǁfit__mutmut_31': xǁScipyFitterǁfit__mutmut_31, 
        'xǁScipyFitterǁfit__mutmut_32': xǁScipyFitterǁfit__mutmut_32, 
        'xǁScipyFitterǁfit__mutmut_33': xǁScipyFitterǁfit__mutmut_33, 
        'xǁScipyFitterǁfit__mutmut_34': xǁScipyFitterǁfit__mutmut_34, 
        'xǁScipyFitterǁfit__mutmut_35': xǁScipyFitterǁfit__mutmut_35, 
        'xǁScipyFitterǁfit__mutmut_36': xǁScipyFitterǁfit__mutmut_36, 
        'xǁScipyFitterǁfit__mutmut_37': xǁScipyFitterǁfit__mutmut_37, 
        'xǁScipyFitterǁfit__mutmut_38': xǁScipyFitterǁfit__mutmut_38, 
        'xǁScipyFitterǁfit__mutmut_39': xǁScipyFitterǁfit__mutmut_39, 
        'xǁScipyFitterǁfit__mutmut_40': xǁScipyFitterǁfit__mutmut_40, 
        'xǁScipyFitterǁfit__mutmut_41': xǁScipyFitterǁfit__mutmut_41, 
        'xǁScipyFitterǁfit__mutmut_42': xǁScipyFitterǁfit__mutmut_42, 
        'xǁScipyFitterǁfit__mutmut_43': xǁScipyFitterǁfit__mutmut_43, 
        'xǁScipyFitterǁfit__mutmut_44': xǁScipyFitterǁfit__mutmut_44, 
        'xǁScipyFitterǁfit__mutmut_45': xǁScipyFitterǁfit__mutmut_45, 
        'xǁScipyFitterǁfit__mutmut_46': xǁScipyFitterǁfit__mutmut_46, 
        'xǁScipyFitterǁfit__mutmut_47': xǁScipyFitterǁfit__mutmut_47, 
        'xǁScipyFitterǁfit__mutmut_48': xǁScipyFitterǁfit__mutmut_48, 
        'xǁScipyFitterǁfit__mutmut_49': xǁScipyFitterǁfit__mutmut_49, 
        'xǁScipyFitterǁfit__mutmut_50': xǁScipyFitterǁfit__mutmut_50, 
        'xǁScipyFitterǁfit__mutmut_51': xǁScipyFitterǁfit__mutmut_51, 
        'xǁScipyFitterǁfit__mutmut_52': xǁScipyFitterǁfit__mutmut_52, 
        'xǁScipyFitterǁfit__mutmut_53': xǁScipyFitterǁfit__mutmut_53, 
        'xǁScipyFitterǁfit__mutmut_54': xǁScipyFitterǁfit__mutmut_54, 
        'xǁScipyFitterǁfit__mutmut_55': xǁScipyFitterǁfit__mutmut_55, 
        'xǁScipyFitterǁfit__mutmut_56': xǁScipyFitterǁfit__mutmut_56, 
        'xǁScipyFitterǁfit__mutmut_57': xǁScipyFitterǁfit__mutmut_57, 
        'xǁScipyFitterǁfit__mutmut_58': xǁScipyFitterǁfit__mutmut_58, 
        'xǁScipyFitterǁfit__mutmut_59': xǁScipyFitterǁfit__mutmut_59, 
        'xǁScipyFitterǁfit__mutmut_60': xǁScipyFitterǁfit__mutmut_60, 
        'xǁScipyFitterǁfit__mutmut_61': xǁScipyFitterǁfit__mutmut_61, 
        'xǁScipyFitterǁfit__mutmut_62': xǁScipyFitterǁfit__mutmut_62, 
        'xǁScipyFitterǁfit__mutmut_63': xǁScipyFitterǁfit__mutmut_63, 
        'xǁScipyFitterǁfit__mutmut_64': xǁScipyFitterǁfit__mutmut_64, 
        'xǁScipyFitterǁfit__mutmut_65': xǁScipyFitterǁfit__mutmut_65, 
        'xǁScipyFitterǁfit__mutmut_66': xǁScipyFitterǁfit__mutmut_66, 
        'xǁScipyFitterǁfit__mutmut_67': xǁScipyFitterǁfit__mutmut_67, 
        'xǁScipyFitterǁfit__mutmut_68': xǁScipyFitterǁfit__mutmut_68, 
        'xǁScipyFitterǁfit__mutmut_69': xǁScipyFitterǁfit__mutmut_69, 
        'xǁScipyFitterǁfit__mutmut_70': xǁScipyFitterǁfit__mutmut_70, 
        'xǁScipyFitterǁfit__mutmut_71': xǁScipyFitterǁfit__mutmut_71, 
        'xǁScipyFitterǁfit__mutmut_72': xǁScipyFitterǁfit__mutmut_72, 
        'xǁScipyFitterǁfit__mutmut_73': xǁScipyFitterǁfit__mutmut_73, 
        'xǁScipyFitterǁfit__mutmut_74': xǁScipyFitterǁfit__mutmut_74, 
        'xǁScipyFitterǁfit__mutmut_75': xǁScipyFitterǁfit__mutmut_75, 
        'xǁScipyFitterǁfit__mutmut_76': xǁScipyFitterǁfit__mutmut_76, 
        'xǁScipyFitterǁfit__mutmut_77': xǁScipyFitterǁfit__mutmut_77, 
        'xǁScipyFitterǁfit__mutmut_78': xǁScipyFitterǁfit__mutmut_78, 
        'xǁScipyFitterǁfit__mutmut_79': xǁScipyFitterǁfit__mutmut_79, 
        'xǁScipyFitterǁfit__mutmut_80': xǁScipyFitterǁfit__mutmut_80, 
        'xǁScipyFitterǁfit__mutmut_81': xǁScipyFitterǁfit__mutmut_81, 
        'xǁScipyFitterǁfit__mutmut_82': xǁScipyFitterǁfit__mutmut_82
    }
    xǁScipyFitterǁfit__mutmut_orig.__name__ = 'xǁScipyFitterǁfit'
