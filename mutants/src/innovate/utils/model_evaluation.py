from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from innovate.base.base import DiffusionModel

from .metrics import (
    calculate_aic,
    calculate_bic,
    calculate_mae,
    calculate_mape,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
    calculate_rss,
    calculate_smape,
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


def model_aic(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    args = [model, t, y]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_model_aic__mutmut_orig, x_model_aic__mutmut_mutants, args, kwargs, None)


def x_model_aic__mutmut_orig(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_1(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_2(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError(None)
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_3(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_4(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("model has not been fitted yet. call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_5(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_6(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = None
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_7(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(None)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_8(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = None
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_9(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(None, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_10(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, None)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_11(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_12(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, )
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_13(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = None
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_14(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = None
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_15(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) - 1
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_16(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 2
    return calculate_aic(n_params, n_samples, rss)


def x_model_aic__mutmut_17(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(None, n_samples, rss)


def x_model_aic__mutmut_18(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, None, rss)


def x_model_aic__mutmut_19(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, None)


def x_model_aic__mutmut_20(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_samples, rss)


def x_model_aic__mutmut_21(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, rss)


def x_model_aic__mutmut_22(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, )

x_model_aic__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_model_aic__mutmut_1': x_model_aic__mutmut_1, 
    'x_model_aic__mutmut_2': x_model_aic__mutmut_2, 
    'x_model_aic__mutmut_3': x_model_aic__mutmut_3, 
    'x_model_aic__mutmut_4': x_model_aic__mutmut_4, 
    'x_model_aic__mutmut_5': x_model_aic__mutmut_5, 
    'x_model_aic__mutmut_6': x_model_aic__mutmut_6, 
    'x_model_aic__mutmut_7': x_model_aic__mutmut_7, 
    'x_model_aic__mutmut_8': x_model_aic__mutmut_8, 
    'x_model_aic__mutmut_9': x_model_aic__mutmut_9, 
    'x_model_aic__mutmut_10': x_model_aic__mutmut_10, 
    'x_model_aic__mutmut_11': x_model_aic__mutmut_11, 
    'x_model_aic__mutmut_12': x_model_aic__mutmut_12, 
    'x_model_aic__mutmut_13': x_model_aic__mutmut_13, 
    'x_model_aic__mutmut_14': x_model_aic__mutmut_14, 
    'x_model_aic__mutmut_15': x_model_aic__mutmut_15, 
    'x_model_aic__mutmut_16': x_model_aic__mutmut_16, 
    'x_model_aic__mutmut_17': x_model_aic__mutmut_17, 
    'x_model_aic__mutmut_18': x_model_aic__mutmut_18, 
    'x_model_aic__mutmut_19': x_model_aic__mutmut_19, 
    'x_model_aic__mutmut_20': x_model_aic__mutmut_20, 
    'x_model_aic__mutmut_21': x_model_aic__mutmut_21, 
    'x_model_aic__mutmut_22': x_model_aic__mutmut_22
}
x_model_aic__mutmut_orig.__name__ = 'x_model_aic'


def model_bic(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    args = [model, t, y]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_model_bic__mutmut_orig, x_model_bic__mutmut_mutants, args, kwargs, None)


def x_model_bic__mutmut_orig(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_1(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_2(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError(None)
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_3(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_4(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("model has not been fitted yet. call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_5(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_6(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = None
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_7(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(None)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_8(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = None
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_9(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(None, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_10(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, None)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_11(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_12(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, )
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_13(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = None
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_14(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = None
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_15(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) - 1
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_16(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 2
    return calculate_bic(n_params, n_samples, rss)


def x_model_bic__mutmut_17(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(None, n_samples, rss)


def x_model_bic__mutmut_18(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, None, rss)


def x_model_bic__mutmut_19(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, None)


def x_model_bic__mutmut_20(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_samples, rss)


def x_model_bic__mutmut_21(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, rss)


def x_model_bic__mutmut_22(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, )

x_model_bic__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_model_bic__mutmut_1': x_model_bic__mutmut_1, 
    'x_model_bic__mutmut_2': x_model_bic__mutmut_2, 
    'x_model_bic__mutmut_3': x_model_bic__mutmut_3, 
    'x_model_bic__mutmut_4': x_model_bic__mutmut_4, 
    'x_model_bic__mutmut_5': x_model_bic__mutmut_5, 
    'x_model_bic__mutmut_6': x_model_bic__mutmut_6, 
    'x_model_bic__mutmut_7': x_model_bic__mutmut_7, 
    'x_model_bic__mutmut_8': x_model_bic__mutmut_8, 
    'x_model_bic__mutmut_9': x_model_bic__mutmut_9, 
    'x_model_bic__mutmut_10': x_model_bic__mutmut_10, 
    'x_model_bic__mutmut_11': x_model_bic__mutmut_11, 
    'x_model_bic__mutmut_12': x_model_bic__mutmut_12, 
    'x_model_bic__mutmut_13': x_model_bic__mutmut_13, 
    'x_model_bic__mutmut_14': x_model_bic__mutmut_14, 
    'x_model_bic__mutmut_15': x_model_bic__mutmut_15, 
    'x_model_bic__mutmut_16': x_model_bic__mutmut_16, 
    'x_model_bic__mutmut_17': x_model_bic__mutmut_17, 
    'x_model_bic__mutmut_18': x_model_bic__mutmut_18, 
    'x_model_bic__mutmut_19': x_model_bic__mutmut_19, 
    'x_model_bic__mutmut_20': x_model_bic__mutmut_20, 
    'x_model_bic__mutmut_21': x_model_bic__mutmut_21, 
    'x_model_bic__mutmut_22': x_model_bic__mutmut_22
}
x_model_bic__mutmut_orig.__name__ = 'x_model_bic'


def get_fit_metrics(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    args = [model, t, y]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_get_fit_metrics__mutmut_orig, x_get_fit_metrics__mutmut_mutants, args, kwargs, None)


def x_get_fit_metrics__mutmut_orig(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_1(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_2(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError(None)

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_3(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_4(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("model has not been fitted yet. call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_5(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_6(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = None

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_7(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(None)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_8(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = None
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_9(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = None

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_10(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) - 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_11(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 2

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_12(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = None

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_13(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(None, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_14(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, None)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_15(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_16(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, )

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_17(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = None
    return metrics


def x_get_fit_metrics__mutmut_18(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "XXMSEXX": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_19(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "mse": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_20(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(None, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_21(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, None),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_22(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_23(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, ),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_24(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "XXRMSEXX": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_25(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "rmse": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_26(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(None, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_27(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, None),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_28(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_29(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, ),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_30(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "XXMAEXX": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_31(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "mae": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_32(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(None, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_33(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, None),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_34(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_35(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, ),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_36(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "XXR-squaredXX": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_37(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "r-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_38(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-SQUARED": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_39(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(None, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_40(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, None),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_41(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_42(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, ),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_43(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "XXMAPEXX": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_44(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "mape": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_45(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(None, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_46(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, None),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_47(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_48(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, ),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_49(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "XXSMAPEXX": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_50(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "smape": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_51(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(None, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_52(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, None),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_53(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_54(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, ),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_55(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "XXRSSXX": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_56(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "rss": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_57(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "XXAICXX": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_58(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "aic": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_59(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(None, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_60(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, None, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_61(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, None),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_62(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_63(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_64(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, ),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_65(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "XXBICXX": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_66(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "bic": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_67(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(None, n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_68(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, None, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_69(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, None),
    }
    return metrics


def x_get_fit_metrics__mutmut_70(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_samples, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_71(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, rss),
    }
    return metrics


def x_get_fit_metrics__mutmut_72(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Calculates various goodness-of-fit metrics for a model.

    Args:
    ----
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns
    -------
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, ),
    }
    return metrics

x_get_fit_metrics__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_get_fit_metrics__mutmut_1': x_get_fit_metrics__mutmut_1, 
    'x_get_fit_metrics__mutmut_2': x_get_fit_metrics__mutmut_2, 
    'x_get_fit_metrics__mutmut_3': x_get_fit_metrics__mutmut_3, 
    'x_get_fit_metrics__mutmut_4': x_get_fit_metrics__mutmut_4, 
    'x_get_fit_metrics__mutmut_5': x_get_fit_metrics__mutmut_5, 
    'x_get_fit_metrics__mutmut_6': x_get_fit_metrics__mutmut_6, 
    'x_get_fit_metrics__mutmut_7': x_get_fit_metrics__mutmut_7, 
    'x_get_fit_metrics__mutmut_8': x_get_fit_metrics__mutmut_8, 
    'x_get_fit_metrics__mutmut_9': x_get_fit_metrics__mutmut_9, 
    'x_get_fit_metrics__mutmut_10': x_get_fit_metrics__mutmut_10, 
    'x_get_fit_metrics__mutmut_11': x_get_fit_metrics__mutmut_11, 
    'x_get_fit_metrics__mutmut_12': x_get_fit_metrics__mutmut_12, 
    'x_get_fit_metrics__mutmut_13': x_get_fit_metrics__mutmut_13, 
    'x_get_fit_metrics__mutmut_14': x_get_fit_metrics__mutmut_14, 
    'x_get_fit_metrics__mutmut_15': x_get_fit_metrics__mutmut_15, 
    'x_get_fit_metrics__mutmut_16': x_get_fit_metrics__mutmut_16, 
    'x_get_fit_metrics__mutmut_17': x_get_fit_metrics__mutmut_17, 
    'x_get_fit_metrics__mutmut_18': x_get_fit_metrics__mutmut_18, 
    'x_get_fit_metrics__mutmut_19': x_get_fit_metrics__mutmut_19, 
    'x_get_fit_metrics__mutmut_20': x_get_fit_metrics__mutmut_20, 
    'x_get_fit_metrics__mutmut_21': x_get_fit_metrics__mutmut_21, 
    'x_get_fit_metrics__mutmut_22': x_get_fit_metrics__mutmut_22, 
    'x_get_fit_metrics__mutmut_23': x_get_fit_metrics__mutmut_23, 
    'x_get_fit_metrics__mutmut_24': x_get_fit_metrics__mutmut_24, 
    'x_get_fit_metrics__mutmut_25': x_get_fit_metrics__mutmut_25, 
    'x_get_fit_metrics__mutmut_26': x_get_fit_metrics__mutmut_26, 
    'x_get_fit_metrics__mutmut_27': x_get_fit_metrics__mutmut_27, 
    'x_get_fit_metrics__mutmut_28': x_get_fit_metrics__mutmut_28, 
    'x_get_fit_metrics__mutmut_29': x_get_fit_metrics__mutmut_29, 
    'x_get_fit_metrics__mutmut_30': x_get_fit_metrics__mutmut_30, 
    'x_get_fit_metrics__mutmut_31': x_get_fit_metrics__mutmut_31, 
    'x_get_fit_metrics__mutmut_32': x_get_fit_metrics__mutmut_32, 
    'x_get_fit_metrics__mutmut_33': x_get_fit_metrics__mutmut_33, 
    'x_get_fit_metrics__mutmut_34': x_get_fit_metrics__mutmut_34, 
    'x_get_fit_metrics__mutmut_35': x_get_fit_metrics__mutmut_35, 
    'x_get_fit_metrics__mutmut_36': x_get_fit_metrics__mutmut_36, 
    'x_get_fit_metrics__mutmut_37': x_get_fit_metrics__mutmut_37, 
    'x_get_fit_metrics__mutmut_38': x_get_fit_metrics__mutmut_38, 
    'x_get_fit_metrics__mutmut_39': x_get_fit_metrics__mutmut_39, 
    'x_get_fit_metrics__mutmut_40': x_get_fit_metrics__mutmut_40, 
    'x_get_fit_metrics__mutmut_41': x_get_fit_metrics__mutmut_41, 
    'x_get_fit_metrics__mutmut_42': x_get_fit_metrics__mutmut_42, 
    'x_get_fit_metrics__mutmut_43': x_get_fit_metrics__mutmut_43, 
    'x_get_fit_metrics__mutmut_44': x_get_fit_metrics__mutmut_44, 
    'x_get_fit_metrics__mutmut_45': x_get_fit_metrics__mutmut_45, 
    'x_get_fit_metrics__mutmut_46': x_get_fit_metrics__mutmut_46, 
    'x_get_fit_metrics__mutmut_47': x_get_fit_metrics__mutmut_47, 
    'x_get_fit_metrics__mutmut_48': x_get_fit_metrics__mutmut_48, 
    'x_get_fit_metrics__mutmut_49': x_get_fit_metrics__mutmut_49, 
    'x_get_fit_metrics__mutmut_50': x_get_fit_metrics__mutmut_50, 
    'x_get_fit_metrics__mutmut_51': x_get_fit_metrics__mutmut_51, 
    'x_get_fit_metrics__mutmut_52': x_get_fit_metrics__mutmut_52, 
    'x_get_fit_metrics__mutmut_53': x_get_fit_metrics__mutmut_53, 
    'x_get_fit_metrics__mutmut_54': x_get_fit_metrics__mutmut_54, 
    'x_get_fit_metrics__mutmut_55': x_get_fit_metrics__mutmut_55, 
    'x_get_fit_metrics__mutmut_56': x_get_fit_metrics__mutmut_56, 
    'x_get_fit_metrics__mutmut_57': x_get_fit_metrics__mutmut_57, 
    'x_get_fit_metrics__mutmut_58': x_get_fit_metrics__mutmut_58, 
    'x_get_fit_metrics__mutmut_59': x_get_fit_metrics__mutmut_59, 
    'x_get_fit_metrics__mutmut_60': x_get_fit_metrics__mutmut_60, 
    'x_get_fit_metrics__mutmut_61': x_get_fit_metrics__mutmut_61, 
    'x_get_fit_metrics__mutmut_62': x_get_fit_metrics__mutmut_62, 
    'x_get_fit_metrics__mutmut_63': x_get_fit_metrics__mutmut_63, 
    'x_get_fit_metrics__mutmut_64': x_get_fit_metrics__mutmut_64, 
    'x_get_fit_metrics__mutmut_65': x_get_fit_metrics__mutmut_65, 
    'x_get_fit_metrics__mutmut_66': x_get_fit_metrics__mutmut_66, 
    'x_get_fit_metrics__mutmut_67': x_get_fit_metrics__mutmut_67, 
    'x_get_fit_metrics__mutmut_68': x_get_fit_metrics__mutmut_68, 
    'x_get_fit_metrics__mutmut_69': x_get_fit_metrics__mutmut_69, 
    'x_get_fit_metrics__mutmut_70': x_get_fit_metrics__mutmut_70, 
    'x_get_fit_metrics__mutmut_71': x_get_fit_metrics__mutmut_71, 
    'x_get_fit_metrics__mutmut_72': x_get_fit_metrics__mutmut_72
}
x_get_fit_metrics__mutmut_orig.__name__ = 'x_get_fit_metrics'


def compare_models(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    args = [models, t_true, y_true]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_compare_models__mutmut_orig, x_compare_models__mutmut_mutants, args, kwargs, None)


def x_compare_models__mutmut_orig(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_1(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = None
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_2(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") and not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_3(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_4(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(None, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_5(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, None) or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_6(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr("predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_7(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, ) or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_8(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "XXpredictXX") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_9(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "PREDICT") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_10(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_11(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(None):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_12(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                None,
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_13(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            break

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_14(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = None
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_15(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(None, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_16(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, None, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_17(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, None)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_18(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_19(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_20(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, )
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_21(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = None
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_22(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["XXParametersXX"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_23(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_24(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["PARAMETERS"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_25(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = None
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_26(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["XXModelXX"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_27(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_28(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["MODEL"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_29(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(None)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_30(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(None)
            continue

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_31(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            break

    return pd.DataFrame(results).set_index("Model")


def x_compare_models__mutmut_32(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index(None)


def x_compare_models__mutmut_33(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(None).set_index("Model")


def x_compare_models__mutmut_34(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("XXModelXX")


def x_compare_models__mutmut_35(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("model")


def x_compare_models__mutmut_36(
    models: dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float],
) -> pd.DataFrame:
    """Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
    ----
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns
    -------
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping.",
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("MODEL")

x_compare_models__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_compare_models__mutmut_1': x_compare_models__mutmut_1, 
    'x_compare_models__mutmut_2': x_compare_models__mutmut_2, 
    'x_compare_models__mutmut_3': x_compare_models__mutmut_3, 
    'x_compare_models__mutmut_4': x_compare_models__mutmut_4, 
    'x_compare_models__mutmut_5': x_compare_models__mutmut_5, 
    'x_compare_models__mutmut_6': x_compare_models__mutmut_6, 
    'x_compare_models__mutmut_7': x_compare_models__mutmut_7, 
    'x_compare_models__mutmut_8': x_compare_models__mutmut_8, 
    'x_compare_models__mutmut_9': x_compare_models__mutmut_9, 
    'x_compare_models__mutmut_10': x_compare_models__mutmut_10, 
    'x_compare_models__mutmut_11': x_compare_models__mutmut_11, 
    'x_compare_models__mutmut_12': x_compare_models__mutmut_12, 
    'x_compare_models__mutmut_13': x_compare_models__mutmut_13, 
    'x_compare_models__mutmut_14': x_compare_models__mutmut_14, 
    'x_compare_models__mutmut_15': x_compare_models__mutmut_15, 
    'x_compare_models__mutmut_16': x_compare_models__mutmut_16, 
    'x_compare_models__mutmut_17': x_compare_models__mutmut_17, 
    'x_compare_models__mutmut_18': x_compare_models__mutmut_18, 
    'x_compare_models__mutmut_19': x_compare_models__mutmut_19, 
    'x_compare_models__mutmut_20': x_compare_models__mutmut_20, 
    'x_compare_models__mutmut_21': x_compare_models__mutmut_21, 
    'x_compare_models__mutmut_22': x_compare_models__mutmut_22, 
    'x_compare_models__mutmut_23': x_compare_models__mutmut_23, 
    'x_compare_models__mutmut_24': x_compare_models__mutmut_24, 
    'x_compare_models__mutmut_25': x_compare_models__mutmut_25, 
    'x_compare_models__mutmut_26': x_compare_models__mutmut_26, 
    'x_compare_models__mutmut_27': x_compare_models__mutmut_27, 
    'x_compare_models__mutmut_28': x_compare_models__mutmut_28, 
    'x_compare_models__mutmut_29': x_compare_models__mutmut_29, 
    'x_compare_models__mutmut_30': x_compare_models__mutmut_30, 
    'x_compare_models__mutmut_31': x_compare_models__mutmut_31, 
    'x_compare_models__mutmut_32': x_compare_models__mutmut_32, 
    'x_compare_models__mutmut_33': x_compare_models__mutmut_33, 
    'x_compare_models__mutmut_34': x_compare_models__mutmut_34, 
    'x_compare_models__mutmut_35': x_compare_models__mutmut_35, 
    'x_compare_models__mutmut_36': x_compare_models__mutmut_36
}
x_compare_models__mutmut_orig.__name__ = 'x_compare_models'


def find_best_model(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    args = [comparison_df, metric, minimize]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_find_best_model__mutmut_orig, x_find_best_model__mutmut_mutants, args, kwargs, None)


def x_find_best_model__mutmut_orig(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_1(
    comparison_df: pd.DataFrame,
    metric: str = "XXRMSEXX",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_2(
    comparison_df: pd.DataFrame,
    metric: str = "rmse",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_3(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_4(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_5(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            None,
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_6(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = None
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def x_find_best_model__mutmut_7(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    minimize: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Identifies the best performing model from a comparison DataFrame.

    Args:
    ----
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns
    -------
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns.",
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = None

    return best_model_row.name, best_model_row.to_dict()

x_find_best_model__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_find_best_model__mutmut_1': x_find_best_model__mutmut_1, 
    'x_find_best_model__mutmut_2': x_find_best_model__mutmut_2, 
    'x_find_best_model__mutmut_3': x_find_best_model__mutmut_3, 
    'x_find_best_model__mutmut_4': x_find_best_model__mutmut_4, 
    'x_find_best_model__mutmut_5': x_find_best_model__mutmut_5, 
    'x_find_best_model__mutmut_6': x_find_best_model__mutmut_6, 
    'x_find_best_model__mutmut_7': x_find_best_model__mutmut_7
}
x_find_best_model__mutmut_orig.__name__ = 'x_find_best_model'


def compute_residuals(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    args = [model, t, y]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_compute_residuals__mutmut_orig, x_compute_residuals__mutmut_mutants, args, kwargs, None)


def x_compute_residuals__mutmut_orig(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_1(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_2(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError(None)

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_3(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_4(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("model has not been fitted yet. call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_5(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_6(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = None
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_7(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(None)
    return np.asarray(y) - np.asarray(y_pred)


def x_compute_residuals__mutmut_8(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(y) + np.asarray(y_pred)


def x_compute_residuals__mutmut_9(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(None) - np.asarray(y_pred)


def x_compute_residuals__mutmut_10(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(None)

x_compute_residuals__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_compute_residuals__mutmut_1': x_compute_residuals__mutmut_1, 
    'x_compute_residuals__mutmut_2': x_compute_residuals__mutmut_2, 
    'x_compute_residuals__mutmut_3': x_compute_residuals__mutmut_3, 
    'x_compute_residuals__mutmut_4': x_compute_residuals__mutmut_4, 
    'x_compute_residuals__mutmut_5': x_compute_residuals__mutmut_5, 
    'x_compute_residuals__mutmut_6': x_compute_residuals__mutmut_6, 
    'x_compute_residuals__mutmut_7': x_compute_residuals__mutmut_7, 
    'x_compute_residuals__mutmut_8': x_compute_residuals__mutmut_8, 
    'x_compute_residuals__mutmut_9': x_compute_residuals__mutmut_9, 
    'x_compute_residuals__mutmut_10': x_compute_residuals__mutmut_10
}
x_compute_residuals__mutmut_orig.__name__ = 'x_compute_residuals'


def residual_acf(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    args = [model, t, y, nlags]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_residual_acf__mutmut_orig, x_residual_acf__mutmut_mutants, args, kwargs, None)


def x_residual_acf__mutmut_orig(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_1(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 41,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_2(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = None
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_3(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(None, t, y)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_4(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, None, y)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_5(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, None)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_6(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(t, y)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_7(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, y)
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_8(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, )
    return acf(residuals, nlags=nlags)


def x_residual_acf__mutmut_9(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(None, nlags=nlags)


def x_residual_acf__mutmut_10(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(residuals, nlags=None)


def x_residual_acf__mutmut_11(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(nlags=nlags)


def x_residual_acf__mutmut_12(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(residuals, )

x_residual_acf__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_residual_acf__mutmut_1': x_residual_acf__mutmut_1, 
    'x_residual_acf__mutmut_2': x_residual_acf__mutmut_2, 
    'x_residual_acf__mutmut_3': x_residual_acf__mutmut_3, 
    'x_residual_acf__mutmut_4': x_residual_acf__mutmut_4, 
    'x_residual_acf__mutmut_5': x_residual_acf__mutmut_5, 
    'x_residual_acf__mutmut_6': x_residual_acf__mutmut_6, 
    'x_residual_acf__mutmut_7': x_residual_acf__mutmut_7, 
    'x_residual_acf__mutmut_8': x_residual_acf__mutmut_8, 
    'x_residual_acf__mutmut_9': x_residual_acf__mutmut_9, 
    'x_residual_acf__mutmut_10': x_residual_acf__mutmut_10, 
    'x_residual_acf__mutmut_11': x_residual_acf__mutmut_11, 
    'x_residual_acf__mutmut_12': x_residual_acf__mutmut_12
}
x_residual_acf__mutmut_orig.__name__ = 'x_residual_acf'


def residual_pacf(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    args = [model, t, y, nlags]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_residual_pacf__mutmut_orig, x_residual_pacf__mutmut_mutants, args, kwargs, None)


def x_residual_pacf__mutmut_orig(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_1(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 41,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_2(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = None
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_3(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(None, t, y)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_4(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, None, y)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_5(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, None)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_6(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(t, y)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_7(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, y)
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_8(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, )
    return pacf(residuals, nlags=nlags)


def x_residual_pacf__mutmut_9(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(None, nlags=nlags)


def x_residual_pacf__mutmut_10(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(residuals, nlags=None)


def x_residual_pacf__mutmut_11(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(nlags=nlags)


def x_residual_pacf__mutmut_12(
    model: DiffusionModel,
    t: Sequence[float],
    y: Sequence[float],
    nlags: int = 40,
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(residuals, )

x_residual_pacf__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_residual_pacf__mutmut_1': x_residual_pacf__mutmut_1, 
    'x_residual_pacf__mutmut_2': x_residual_pacf__mutmut_2, 
    'x_residual_pacf__mutmut_3': x_residual_pacf__mutmut_3, 
    'x_residual_pacf__mutmut_4': x_residual_pacf__mutmut_4, 
    'x_residual_pacf__mutmut_5': x_residual_pacf__mutmut_5, 
    'x_residual_pacf__mutmut_6': x_residual_pacf__mutmut_6, 
    'x_residual_pacf__mutmut_7': x_residual_pacf__mutmut_7, 
    'x_residual_pacf__mutmut_8': x_residual_pacf__mutmut_8, 
    'x_residual_pacf__mutmut_9': x_residual_pacf__mutmut_9, 
    'x_residual_pacf__mutmut_10': x_residual_pacf__mutmut_10, 
    'x_residual_pacf__mutmut_11': x_residual_pacf__mutmut_11, 
    'x_residual_pacf__mutmut_12': x_residual_pacf__mutmut_12
}
x_residual_pacf__mutmut_orig.__name__ = 'x_residual_pacf'
