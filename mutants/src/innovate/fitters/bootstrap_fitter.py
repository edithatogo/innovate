from collections.abc import Sequence
from typing import Any

import numpy as np

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


class BootstrapFitter:
    """A fitter class that uses bootstrapping to estimate parameter uncertainty."""

    def __init__(self, fitter: Any, n_bootstraps: int = 100):
        args = [fitter, n_bootstraps]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBootstrapFitterǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁBootstrapFitterǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁBootstrapFitterǁ__init____mutmut_orig(self, fitter: Any, n_bootstraps: int = 100):
        self.fitter = fitter
        self.n_bootstraps = n_bootstraps
        self.bootstrapped_params: list[dict[str, float]] = []

    def xǁBootstrapFitterǁ__init____mutmut_1(self, fitter: Any, n_bootstraps: int = 101):
        self.fitter = fitter
        self.n_bootstraps = n_bootstraps
        self.bootstrapped_params: list[dict[str, float]] = []

    def xǁBootstrapFitterǁ__init____mutmut_2(self, fitter: Any, n_bootstraps: int = 100):
        self.fitter = None
        self.n_bootstraps = n_bootstraps
        self.bootstrapped_params: list[dict[str, float]] = []

    def xǁBootstrapFitterǁ__init____mutmut_3(self, fitter: Any, n_bootstraps: int = 100):
        self.fitter = fitter
        self.n_bootstraps = None
        self.bootstrapped_params: list[dict[str, float]] = []

    def xǁBootstrapFitterǁ__init____mutmut_4(self, fitter: Any, n_bootstraps: int = 100):
        self.fitter = fitter
        self.n_bootstraps = n_bootstraps
        self.bootstrapped_params: list[dict[str, float]] = None
    
    xǁBootstrapFitterǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBootstrapFitterǁ__init____mutmut_1': xǁBootstrapFitterǁ__init____mutmut_1, 
        'xǁBootstrapFitterǁ__init____mutmut_2': xǁBootstrapFitterǁ__init____mutmut_2, 
        'xǁBootstrapFitterǁ__init____mutmut_3': xǁBootstrapFitterǁ__init____mutmut_3, 
        'xǁBootstrapFitterǁ__init____mutmut_4': xǁBootstrapFitterǁ__init____mutmut_4
    }
    xǁBootstrapFitterǁ__init____mutmut_orig.__name__ = 'xǁBootstrapFitterǁ__init__'

    def fit(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        args = [model, t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBootstrapFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁBootstrapFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁBootstrapFitterǁfit__mutmut_orig(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_1(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = None
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_2(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(None)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_3(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = None
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_4(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(None)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_5(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = None

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_6(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(None):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_7(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = None
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_8(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(None, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_9(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, None, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_10(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=None)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_11(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_12(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_13(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, )
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_14(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=False)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_15(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = None
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_16(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = None

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_17(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = None

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_18(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(None)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_19(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(None, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_20(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, None, y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_21(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), None, **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_22(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_23(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_24(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_25(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), )
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_26(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(None)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                continue

    def xǁBootstrapFitterǁfit__mutmut_27(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(None)
                continue

    def xǁBootstrapFitterǁfit__mutmut_28(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> None:
        t_arr = np.array(t)
        y_arr = np.array(y)
        n_samples = len(t_arr)

        for _ in range(self.n_bootstraps):
            # Resample data with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            t_resampled = t_arr[indices]
            y_resampled = y_arr[indices]

            # Create a new model instance for each bootstrap iteration
            # This is important to avoid parameter contamination between iterations
            boot_model = type(model)()

            try:
                boot_model.fit(self.fitter, t_resampled.tolist(), y_resampled.tolist(), **kwargs)
                self.bootstrapped_params.append(boot_model.params_)
            except RuntimeError as e:
                # Handle cases where fitting might fail for a resampled dataset
                print(f"Warning: Fitting failed for a bootstrap sample: {e}")
                break
    
    xǁBootstrapFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBootstrapFitterǁfit__mutmut_1': xǁBootstrapFitterǁfit__mutmut_1, 
        'xǁBootstrapFitterǁfit__mutmut_2': xǁBootstrapFitterǁfit__mutmut_2, 
        'xǁBootstrapFitterǁfit__mutmut_3': xǁBootstrapFitterǁfit__mutmut_3, 
        'xǁBootstrapFitterǁfit__mutmut_4': xǁBootstrapFitterǁfit__mutmut_4, 
        'xǁBootstrapFitterǁfit__mutmut_5': xǁBootstrapFitterǁfit__mutmut_5, 
        'xǁBootstrapFitterǁfit__mutmut_6': xǁBootstrapFitterǁfit__mutmut_6, 
        'xǁBootstrapFitterǁfit__mutmut_7': xǁBootstrapFitterǁfit__mutmut_7, 
        'xǁBootstrapFitterǁfit__mutmut_8': xǁBootstrapFitterǁfit__mutmut_8, 
        'xǁBootstrapFitterǁfit__mutmut_9': xǁBootstrapFitterǁfit__mutmut_9, 
        'xǁBootstrapFitterǁfit__mutmut_10': xǁBootstrapFitterǁfit__mutmut_10, 
        'xǁBootstrapFitterǁfit__mutmut_11': xǁBootstrapFitterǁfit__mutmut_11, 
        'xǁBootstrapFitterǁfit__mutmut_12': xǁBootstrapFitterǁfit__mutmut_12, 
        'xǁBootstrapFitterǁfit__mutmut_13': xǁBootstrapFitterǁfit__mutmut_13, 
        'xǁBootstrapFitterǁfit__mutmut_14': xǁBootstrapFitterǁfit__mutmut_14, 
        'xǁBootstrapFitterǁfit__mutmut_15': xǁBootstrapFitterǁfit__mutmut_15, 
        'xǁBootstrapFitterǁfit__mutmut_16': xǁBootstrapFitterǁfit__mutmut_16, 
        'xǁBootstrapFitterǁfit__mutmut_17': xǁBootstrapFitterǁfit__mutmut_17, 
        'xǁBootstrapFitterǁfit__mutmut_18': xǁBootstrapFitterǁfit__mutmut_18, 
        'xǁBootstrapFitterǁfit__mutmut_19': xǁBootstrapFitterǁfit__mutmut_19, 
        'xǁBootstrapFitterǁfit__mutmut_20': xǁBootstrapFitterǁfit__mutmut_20, 
        'xǁBootstrapFitterǁfit__mutmut_21': xǁBootstrapFitterǁfit__mutmut_21, 
        'xǁBootstrapFitterǁfit__mutmut_22': xǁBootstrapFitterǁfit__mutmut_22, 
        'xǁBootstrapFitterǁfit__mutmut_23': xǁBootstrapFitterǁfit__mutmut_23, 
        'xǁBootstrapFitterǁfit__mutmut_24': xǁBootstrapFitterǁfit__mutmut_24, 
        'xǁBootstrapFitterǁfit__mutmut_25': xǁBootstrapFitterǁfit__mutmut_25, 
        'xǁBootstrapFitterǁfit__mutmut_26': xǁBootstrapFitterǁfit__mutmut_26, 
        'xǁBootstrapFitterǁfit__mutmut_27': xǁBootstrapFitterǁfit__mutmut_27, 
        'xǁBootstrapFitterǁfit__mutmut_28': xǁBootstrapFitterǁfit__mutmut_28
    }
    xǁBootstrapFitterǁfit__mutmut_orig.__name__ = 'xǁBootstrapFitterǁfit'

    def get_parameter_estimates(self) -> dict[str, list[float]]:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBootstrapFitterǁget_parameter_estimates__mutmut_orig'), object.__getattribute__(self, 'xǁBootstrapFitterǁget_parameter_estimates__mutmut_mutants'), args, kwargs, self)

    def xǁBootstrapFitterǁget_parameter_estimates__mutmut_orig(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = self.bootstrapped_params[0].keys()
        estimates: dict[str, list[float]] = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def xǁBootstrapFitterǁget_parameter_estimates__mutmut_1(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = self.bootstrapped_params[0].keys()
        estimates: dict[str, list[float]] = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def xǁBootstrapFitterǁget_parameter_estimates__mutmut_2(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = None
        estimates: dict[str, list[float]] = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def xǁBootstrapFitterǁget_parameter_estimates__mutmut_3(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = self.bootstrapped_params[1].keys()
        estimates: dict[str, list[float]] = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def xǁBootstrapFitterǁget_parameter_estimates__mutmut_4(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = self.bootstrapped_params[0].keys()
        estimates: dict[str, list[float]] = None

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(value)
        return estimates

    def xǁBootstrapFitterǁget_parameter_estimates__mutmut_5(self) -> dict[str, list[float]]:
        """Returns a dictionary of parameter names to lists of bootstrapped values."""
        if not self.bootstrapped_params:
            return {}

        # Assuming all models have the same parameter names
        param_names = self.bootstrapped_params[0].keys()
        estimates: dict[str, list[float]] = {name: [] for name in param_names}

        for params_dict in self.bootstrapped_params:
            for name, value in params_dict.items():
                estimates[name].append(None)
        return estimates
    
    xǁBootstrapFitterǁget_parameter_estimates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBootstrapFitterǁget_parameter_estimates__mutmut_1': xǁBootstrapFitterǁget_parameter_estimates__mutmut_1, 
        'xǁBootstrapFitterǁget_parameter_estimates__mutmut_2': xǁBootstrapFitterǁget_parameter_estimates__mutmut_2, 
        'xǁBootstrapFitterǁget_parameter_estimates__mutmut_3': xǁBootstrapFitterǁget_parameter_estimates__mutmut_3, 
        'xǁBootstrapFitterǁget_parameter_estimates__mutmut_4': xǁBootstrapFitterǁget_parameter_estimates__mutmut_4, 
        'xǁBootstrapFitterǁget_parameter_estimates__mutmut_5': xǁBootstrapFitterǁget_parameter_estimates__mutmut_5
    }
    xǁBootstrapFitterǁget_parameter_estimates__mutmut_orig.__name__ = 'xǁBootstrapFitterǁget_parameter_estimates'

    def get_confidence_intervals(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        args = [alpha]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBootstrapFitterǁget_confidence_intervals__mutmut_orig'), object.__getattribute__(self, 'xǁBootstrapFitterǁget_confidence_intervals__mutmut_mutants'), args, kwargs, self)

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_orig(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_1(
        self,
        alpha: float = 1.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_2(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = None
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_3(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = None
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_4(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = None
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_5(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(None, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_6(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, None)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_7(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile((alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_8(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, )
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_9(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) / 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_10(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha * 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_11(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 3) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_12(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 101)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_13(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = None
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_14(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(None, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_15(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, None)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_16(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile((1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_17(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, )
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_18(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) / 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_19(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 + alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_20(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (2 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_21(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha * 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_22(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 3) * 100)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_23(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 101)
                cis[name] = {"lower": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_24(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = None
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_25(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"XXlowerXX": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_26(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"LOWER": float(lower), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_27(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(None), "upper": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_28(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "XXupperXX": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_29(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "UPPER": float(upper)}
        return cis

    def xǁBootstrapFitterǁget_confidence_intervals__mutmut_30(
        self,
        alpha: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Returns confidence intervals for each parameter."""
        estimates = self.get_parameter_estimates()
        cis = {}
        for name, values in estimates.items():
            if values:
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                cis[name] = {"lower": float(lower), "upper": float(None)}
        return cis
    
    xǁBootstrapFitterǁget_confidence_intervals__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBootstrapFitterǁget_confidence_intervals__mutmut_1': xǁBootstrapFitterǁget_confidence_intervals__mutmut_1, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_2': xǁBootstrapFitterǁget_confidence_intervals__mutmut_2, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_3': xǁBootstrapFitterǁget_confidence_intervals__mutmut_3, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_4': xǁBootstrapFitterǁget_confidence_intervals__mutmut_4, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_5': xǁBootstrapFitterǁget_confidence_intervals__mutmut_5, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_6': xǁBootstrapFitterǁget_confidence_intervals__mutmut_6, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_7': xǁBootstrapFitterǁget_confidence_intervals__mutmut_7, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_8': xǁBootstrapFitterǁget_confidence_intervals__mutmut_8, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_9': xǁBootstrapFitterǁget_confidence_intervals__mutmut_9, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_10': xǁBootstrapFitterǁget_confidence_intervals__mutmut_10, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_11': xǁBootstrapFitterǁget_confidence_intervals__mutmut_11, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_12': xǁBootstrapFitterǁget_confidence_intervals__mutmut_12, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_13': xǁBootstrapFitterǁget_confidence_intervals__mutmut_13, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_14': xǁBootstrapFitterǁget_confidence_intervals__mutmut_14, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_15': xǁBootstrapFitterǁget_confidence_intervals__mutmut_15, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_16': xǁBootstrapFitterǁget_confidence_intervals__mutmut_16, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_17': xǁBootstrapFitterǁget_confidence_intervals__mutmut_17, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_18': xǁBootstrapFitterǁget_confidence_intervals__mutmut_18, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_19': xǁBootstrapFitterǁget_confidence_intervals__mutmut_19, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_20': xǁBootstrapFitterǁget_confidence_intervals__mutmut_20, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_21': xǁBootstrapFitterǁget_confidence_intervals__mutmut_21, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_22': xǁBootstrapFitterǁget_confidence_intervals__mutmut_22, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_23': xǁBootstrapFitterǁget_confidence_intervals__mutmut_23, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_24': xǁBootstrapFitterǁget_confidence_intervals__mutmut_24, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_25': xǁBootstrapFitterǁget_confidence_intervals__mutmut_25, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_26': xǁBootstrapFitterǁget_confidence_intervals__mutmut_26, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_27': xǁBootstrapFitterǁget_confidence_intervals__mutmut_27, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_28': xǁBootstrapFitterǁget_confidence_intervals__mutmut_28, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_29': xǁBootstrapFitterǁget_confidence_intervals__mutmut_29, 
        'xǁBootstrapFitterǁget_confidence_intervals__mutmut_30': xǁBootstrapFitterǁget_confidence_intervals__mutmut_30
    }
    xǁBootstrapFitterǁget_confidence_intervals__mutmut_orig.__name__ = 'xǁBootstrapFitterǁget_confidence_intervals'

    def get_standard_errors(self) -> dict[str, float]:
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁBootstrapFitterǁget_standard_errors__mutmut_orig'), object.__getattribute__(self, 'xǁBootstrapFitterǁget_standard_errors__mutmut_mutants'), args, kwargs, self)

    def xǁBootstrapFitterǁget_standard_errors__mutmut_orig(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = float(np.std(values))
        return ses

    def xǁBootstrapFitterǁget_standard_errors__mutmut_1(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = None
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = float(np.std(values))
        return ses

    def xǁBootstrapFitterǁget_standard_errors__mutmut_2(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = None
        for name, values in estimates.items():
            if values:
                ses[name] = float(np.std(values))
        return ses

    def xǁBootstrapFitterǁget_standard_errors__mutmut_3(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = None
        return ses

    def xǁBootstrapFitterǁget_standard_errors__mutmut_4(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = float(None)
        return ses

    def xǁBootstrapFitterǁget_standard_errors__mutmut_5(self) -> dict[str, float]:
        """Returns standard errors for each parameter."""
        estimates = self.get_parameter_estimates()
        ses = {}
        for name, values in estimates.items():
            if values:
                ses[name] = float(np.std(None))
        return ses
    
    xǁBootstrapFitterǁget_standard_errors__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁBootstrapFitterǁget_standard_errors__mutmut_1': xǁBootstrapFitterǁget_standard_errors__mutmut_1, 
        'xǁBootstrapFitterǁget_standard_errors__mutmut_2': xǁBootstrapFitterǁget_standard_errors__mutmut_2, 
        'xǁBootstrapFitterǁget_standard_errors__mutmut_3': xǁBootstrapFitterǁget_standard_errors__mutmut_3, 
        'xǁBootstrapFitterǁget_standard_errors__mutmut_4': xǁBootstrapFitterǁget_standard_errors__mutmut_4, 
        'xǁBootstrapFitterǁget_standard_errors__mutmut_5': xǁBootstrapFitterǁget_standard_errors__mutmut_5
    }
    xǁBootstrapFitterǁget_standard_errors__mutmut_orig.__name__ = 'xǁBootstrapFitterǁget_standard_errors'
